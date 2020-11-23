'''
    pytorch implementation of our RMN model
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gumbel as gumbel
from models.allennlp_beamsearch import BeamSearch


# ------------------------------------------------------
# ------------ Soft Attention Mechanism ----------------
# ------------------------------------------------------

class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha


# ------------------------------------------------------
# ------------ Gumbel Attention Mechanism --------------
# ------------------------------------------------------

class GumbelAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(GumbelAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        outputs = self.wa(torch.tanh(inputs)).squeeze(-1)
        if self.training:
            alpha = gumbel.st_gumbel_softmax(outputs)
        else:
            alpha = gumbel.greedy_select(outputs).float()

        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

# ------------------------------------------------------
# --- Pre-process visual features by Bi-LSTM Encoder ---
# ------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.hidden_size
        self.use_multi_gpu = opt.use_multi_gpu

        # frame feature embedding
        self.frame_feature_embed = nn.Linear(opt.a_feature_size, opt.frame_projected_size)
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        self.bi_lstm1 = nn.LSTM(opt.frame_projected_size, opt.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop1 = nn.Dropout(p=opt.dropout)

        # i3d feature embedding
        self.i3d_feature_embed = nn.Linear(opt.m_feature_size, opt.frame_projected_size)
        nn.init.xavier_normal_(self.i3d_feature_embed.weight)
        self.bi_lstm2 = nn.LSTM(opt.frame_projected_size, opt.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop2 = nn.Dropout(p=opt.dropout)

        # region feature embedding
        self.region_feature_embed = nn.Linear(opt.region_feature_size, opt.region_projected_size)
        nn.init.xavier_normal_(self.region_feature_embed.weight)

        # location feature embedding
        self.spatial_feature_embed = nn.Linear(opt.spatial_feature_size, opt.spatial_projected_size)
        nn.init.xavier_normal_(self.spatial_feature_embed.weight)

        # time embedding matrix
        self.time_feats = nn.Parameter(torch.randn(opt.max_frames, opt.time_size))

        # fuse region+loc+time
        in_size = opt.spatial_projected_size + opt.time_size
        self.visual_embed = nn.Linear(opt.region_projected_size + in_size, opt.region_projected_size)
        nn.init.xavier_normal_(self.visual_embed.weight)
        self.visual_drop = nn.Dropout(p=opt.dropout)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, cnn_feats, region_feats, spatial_feats):
        '''
        :param cnn_feats: (batch_size, max_frames, m_feature_size + a_feature_size)
        :param region_feats: (batch_size, max_frames, num_boxes, region_feature_size)
        :param spatial_feats: (batch_size, max_frames, num_boxes, spatial_feature_size)
        :return: output of Bidirectional LSTM and embedded region features
        '''
        # 2d cnn or 3d cnn or 2d+3d cnn
        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        frame_feats = cnn_feats[:, :, :self.a_feature_size].contiguous()
        i3d_feats = cnn_feats[:, :, -self.m_feature_size:].contiguous()

        # flatten parameters if use multiple gpu
        if self.use_multi_gpu:
            self.bi_lstm1.flatten_parameters()
            self.bi_lstm2.flatten_parameters()

        # frame feature embedding
        embedded_frame_feats = self.frame_feature_embed(frame_feats)
        lstm_h1, lstm_c1 = self._init_lstm_state(frame_feats)
        # bidirectional lstm encoder
        frame_feats, _ = self.bi_lstm1(embedded_frame_feats, (lstm_h1, lstm_c1))
        frame_feats = self.lstm_drop1(frame_feats)

        # i3d feature embedding
        embedded_i3d_feats = self.i3d_feature_embed(i3d_feats)
        lstm_h2, lstm_c2 = self._init_lstm_state(i3d_feats)
        # bidirectional lstm encoder
        i3d_feats, _ = self.bi_lstm2(embedded_i3d_feats, (lstm_h2, lstm_c2))
        i3d_feats = self.lstm_drop2(i3d_feats)

        # region feature embedding
        region_feats = self.region_feature_embed(region_feats)
        # spatial feature embedding
        loc_feats = self.spatial_feature_embed(spatial_feats)
        # time feature embedding
        bsz, _, num_boxes, _ = region_feats.size()
        time_feats = self.time_feats.unsqueeze(0).unsqueeze(2).repeat(bsz, 1, num_boxes, 1)
        # object feature
        object_feats = torch.cat([region_feats, loc_feats, time_feats], dim=-1)
        object_feats = self.visual_drop(torch.tanh(self.visual_embed(object_feats)))

        return frame_feats, i3d_feats, object_feats


# ------------------------------------------------------
# -------------------- LOCATE Module -------------------
# ------------------------------------------------------

class LOCATE(nn.Module):
    def __init__(self, opt):
        super(LOCATE, self).__init__()
        # spatial soft attention module
        
        self.spatial_attn = SoftAttention(opt.region_projected_size, opt.hidden_size, opt.att_size)
        # temporal soft attention module
        feat_size = opt.region_projected_size + opt.hidden_size * 2
        self.temp_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)

    def forward(self, frame_feats, object_feats, hidden_state):
        """
        :param frame_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: loc_feat: (batch_size, feat_size)
        """
        # spatial attention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # temporal attention
        feat = torch.cat([object_feats_att, frame_feats], dim=-1)
        loc_feat, _ = self.temp_attn(feat, hidden_state)
        return loc_feat


# ------------------------------------------------------
# -------------------- RELATE Module -------------------
# ------------------------------------------------------

class RELATE(nn.Module):
    def __init__(self, opt):
        super(RELATE, self).__init__()

        # spatial soft attention module
        region_feat_size = opt.region_projected_size
        self.spatial_attn = SoftAttention(region_feat_size, opt.hidden_size, opt.hidden_size)

        # temporal soft attention module
        feat_size = region_feat_size + opt.hidden_size * 2
        self.relation_attn = SoftAttention(2*feat_size, opt.hidden_size, opt.hidden_size)

    def forward(self, i3d_feats, object_feats, hidden_state):
        '''
        :param i3d_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: rel_feat
        '''
        # spatial atttention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # generate pair-wise feature
        feat = torch.cat([object_feats_att, i3d_feats], dim=-1)
        feat1 = feat.repeat(1, max_frames, 1)
        feat2 = feat.repeat(1, 1, max_frames).reshape(bsz, max_frames*max_frames, -1)
        pairwise_feat = torch.cat([feat1, feat2], dim=-1)

        # temporal attention
        rel_feat, _ = self.relation_attn(pairwise_feat, hidden_state)
        return rel_feat


# ------------------------------------------------------
# -------------------- FUNC Module ---------------------
# ------------------------------------------------------

class FUNC(nn.Module):
    def __init__(self, opt):
        super(FUNC, self).__init__()
        self.cell_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, cells, hidden_state):
        '''
        :param cells: previous memory states of decoder LSTM
        :param hidden_state: (batch_size, hidden_size)
        :return: func_feat
        '''
        func_feat, _ = self.cell_attn(cells, hidden_state)
        return func_feat


# ------------------------------------------------------
# ------------------- Module Selector ------------------
# ------------------------------------------------------

class ModuleSelection(nn.Module):
    def __init__(self, opt):
        super(ModuleSelection, self).__init__()
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func

        if opt.use_loc:
            loc_feat_size = opt.region_projected_size + opt.hidden_size * 2
            self.loc_fc = nn.Linear(loc_feat_size, opt.hidden_size)
            nn.init.xavier_normal_(self.loc_fc.weight)

        if opt.use_rel:
            rel_feat_size = 2 * (opt.region_projected_size + 2 * opt.hidden_size)
            self.rel_fc = nn.Linear(rel_feat_size, opt.hidden_size)
            nn.init.xavier_normal_(self.rel_fc.weight)

        if opt.use_func:
            func_size = opt.hidden_size
            self.func_fc = nn.Linear(func_size, opt.hidden_size)
            nn.init.xavier_normal_(self.func_fc.weight)

        if opt.use_loc and opt.use_rel and opt.use_func:
            if opt.attention == 'soft':
                self.module_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            elif opt.attention == 'gumbel':
                self.module_attn = GumbelAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, loc_feats, rel_feats, func_feats, hidden_state):
        '''
        soft attention: Weighted sum of three features
        gumbel attention: Choose one of three features
        '''
        loc_feats = self.loc_fc(loc_feats) if self.use_loc else None
        rel_feats = self.rel_fc(rel_feats) if self.use_rel else None
        func_feats = self.func_fc(func_feats) if self.use_func else None

        if self.use_loc and self.use_rel and self.use_func:
            feats = torch.stack([loc_feats, rel_feats, func_feats], dim=1)
            feats, module_weight = self.module_attn(feats, hidden_state)

        elif self.use_loc and not self.use_rel:
            feats = loc_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()
        elif self.use_rel and not self.use_loc:
            feats = rel_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()

        return feats, module_weight


# ------------------------------------------------------
# --------------- Language LSTM Decoder ----------------
# ------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, opt, vocab):
        super(Decoder, self).__init__()
        self.region_projected_size = opt.region_projected_size
        self.hidden_size = opt.hidden_size
        self.word_size = opt.word_size
        self.max_words = opt.max_words
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = opt.beam_size
        self.use_multi_gpu = opt.use_multi_gpu
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func
        self.batch_size = 32

        # modules
        self.loc = LOCATE(opt)
        self.rel = RELATE(opt)
        self.func = FUNC(opt)
        self.module_selection = ModuleSelection(opt)

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        self.word_drop = nn.Dropout(p=opt.dropout)

        # attention lstm
        visual_feat_size = opt.hidden_size * 4 + opt.region_projected_size
        att_insize = opt.hidden_size + opt.word_size + visual_feat_size
        self.att_lstm = nn.LSTMCell(att_insize, opt.hidden_size)
        self.att_lstm_drop = nn.Dropout(p=opt.dropout)

        # language lstm
        self.lang_lstm = nn.LSTMCell(opt.hidden_size * 2, opt.hidden_size)
        self.lstm_drop = nn.Dropout(p=opt.dropout)

        # final output layer
        self.out_fc = nn.Linear(opt.hidden_size * 3, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.beam_search = BeamSearch(vocab('<end>'), self.max_words, self.beam_size, per_node_beam_size=self.beam_size)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio=1.0):
        self.batch_size = frame_feats.size(0)
        infer = True if captions is None else False

        # visual input of attention lstm
        global_frame_feat = torch.mean(frame_feats, dim=1)
        global_i3d_feat = torch.mean(i3d_feats, dim=1)
        global_object_feat = torch.mean(torch.mean(object_feats, dim=2), dim=1)
        global_feat = torch.cat([global_frame_feat, global_i3d_feat, global_object_feat], dim=1)

        # initialize lstm state
        lang_lstm_h, lang_lstm_c = self._init_lstm_state(global_feat)
        att_lstm_h, att_lstm_c = self._init_lstm_state(global_feat)

        # add a '<start>' sign
        start_id = self.vocab('<start>')
        start_id = global_feat.data.new(global_feat.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)  # b*w

        # training stage
        outputs = []
        previous_cells = []
        previous_cells.append(lang_lstm_c)
        module_weights = []
        if not infer or self.beam_size == 1:
            for i in range(self.max_words):
                if not infer and not self.use_multi_gpu and captions[:, i].data.sum() == 0:
                    break

                # attention lstm
                att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                                       (att_lstm_h, att_lstm_c))
                att_lstm_h = self.att_lstm_drop(att_lstm_h)

                # lstm decoder with attention model
                word_logits, module_weight, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                    previous_cells, att_lstm_h, lang_lstm_h,
                                                                    lang_lstm_c)
                module_weights.append(module_weight)
                previous_cells.append(lang_lstm_c)

                # teacher_forcing: a training trick
                use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    word_id = word_logits.max(1)[1]
                word = self.word_embed(word_id)
                word = self.word_drop(word)

                if infer:
                    outputs.append(word_id)
                else:
                    outputs.append(word_logits)

            outputs = torch.stack(outputs, dim=1)  # b*m*v(train) or b*m(infer)
            module_weights = torch.stack(module_weights, dim=1)
        else:
            # apply beam search if beam size > 1 during testing
            start_state = {'att_lstm_h': att_lstm_h, 'att_lstm_c': att_lstm_c, 'lang_lstm_h': lang_lstm_h,
                           'lang_lstm_c': lang_lstm_c, 'global_feat': global_feat, 'frame_feats': frame_feats,
                           'i3d_feats': i3d_feats, 'object_feats': object_feats, 'previous_cells': previous_cells}
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)
            module_weights = None

        return outputs, module_weights

    def beam_step(self, last_predictions, current_state):
        '''
        A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        '''
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            att_lstm_h = current_state['att_lstm_h'][:, i, :]
            att_lstm_c = current_state['att_lstm_c'][:, i, :]
            lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            global_feat = current_state['global_feat'][:, i, :]
            frame_feats = current_state['frame_feats'][:, i, :]
            i3d_feats = current_state['i3d_feats'][:, i, :]
            object_feats = current_state['object_feats'][:, i, :]
            previous_cells = current_state['previous_cells'][:, i, :]

            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)
            # attention lstm
            att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                                   (att_lstm_h, att_lstm_c))
            att_lstm_h = self.att_lstm_drop(att_lstm_h)

            # language lstm decoder
            word_logits, module_weight, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                               previous_cells, att_lstm_h, lang_lstm_h,
                                                                               lang_lstm_c)
            previous_cells = torch.cat([previous_cells, lang_lstm_c.unsqueeze(1)], dim=1)
            # store log probabilities
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state
            new_state['att_lstm_h'].append(att_lstm_h)
            new_state['att_lstm_c'].append(att_lstm_c)
            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['global_feat'].append(global_feat)
            new_state['frame_feats'].append(frame_feats)
            new_state['i3d_feats'].append(i3d_feats)
            new_state['object_feats'].append(object_feats)
            new_state['previous_cells'].append(previous_cells)

        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

    def decode(self, frame_feats, i3d_feats, object_feats, previous_cells, att_lstm_h, lang_lstm_h, lang_lstm_c):
        if isinstance(previous_cells, list):
            previous_cells = torch.stack(previous_cells, dim=1)

        # LOCATE, RELATE, FUNC modules
        if not self.use_rel and not self.use_loc:
            raise ValueError('use locate or relation, all use both')
        loc_feats = self.loc(frame_feats, object_feats, att_lstm_h) if self.use_loc else None
        rel_feats = self.rel(i3d_feats, object_feats, att_lstm_h) if self.use_rel else None
        func_feats = self.func(previous_cells, att_lstm_h) if self.use_func else None
        feats, module_weight = self.module_selection(loc_feats, rel_feats, func_feats, att_lstm_h)

        # language lstm decoder
        decoder_input = torch.cat([feats, att_lstm_h], dim=1)
        lstm_h, lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
        lstm_h = self.lstm_drop(lstm_h)
        decoder_output = torch.tanh(self.out_fc(torch.cat([lstm_h, decoder_input], dim=1)))
        word_logits = self.word_restore(decoder_output)  # b*v
        return word_logits, module_weight, lstm_h, lstm_c

    def decode_tokens(self, tokens):
        '''
        convert word index to caption
        :param tokens: input word index
        :return: capiton
        '''
        words = []
        for token in tokens:
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        captions = ' '.join(words)
        return captions


# ------------------------------------------------------
# ----------------- Captioning Model -------------------
# ------------------------------------------------------

class CapModel(nn.Module):
    def __init__(self, opt, vocab):
        super(CapModel, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt, vocab)

    def forward(self, cnn_feats, region_feats, spatial_feats, captions, teacher_forcing_ratio=1.0):
        frame_feats, i3d_feats, object_feats = self.encoder(cnn_feats, region_feats, spatial_feats)
        outputs, module_weights = self.decoder(frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio)
        return outputs, module_weights
