from utils.utils import Vocabulary
import models
from utils.opt import parse_opt
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os
import pickle
import torch
import h5py
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(vocab, frame_feat, region_feat, spatial_feat, net, vid):
    outputs, module_weights = net(frame_feat, region_feat, spatial_feat, None)

    words = []
    for i, token in enumerate(outputs.data.squeeze()):
        if token == vocab('<end>'):
            break
        word = vocab.idx2word[token]
        # print(word)
        words.append(word)

    module_weights = module_weights.squeeze().cpu().detach().numpy()[0: len(words), :]
    tag = np.argmax(module_weights, axis=1)

    # LOCATE, RELATE, FUNC words
    loc_words = [w for i, w in enumerate(words[0:]) if tag[i] == 0]
    rel_words = [w for i, w in enumerate(words[0:]) if tag[i] == 1]
    func_words = [w for i, w in enumerate(words[0:]) if tag[i] == 2]

    # visualization

    # plt.figure('visualization')
    # x = np.arange(len(words))
    # plt.bar(x, module_weights[:, 0], alpha=0.9, width=0.2, label='LOCATE')
    # plt.bar(x+0.2, module_weights[:, 1], tick_label=words[0:], alpha=0.9, width=0.2, label='RELATE')
    # plt.bar(x+0.4, module_weights[:, 2], alpha=0.9, width=0.2, label='FUNC')
    # plt.legend()
    # plt.title(vid)
    # plt.show()

    return words, loc_words, rel_words, func_words


def PrintStatistics(split):
    if split == 'train':
        path = opt.train_caption_pkl_path
    elif split == 'test':
        path = opt.test_caption_pkl_path
    with open(path, 'rb') as f:
        _, pos_tags, cap_len, _ = pickle.load(f)

    num_loc, num_rel, num_func = 0, 0, 0
    for i in range(len(pos_tags)):
        pos = Counter(pos_tags[i].tolist()[0 : cap_len[i]])
        num_loc += pos[0]
        num_rel += pos[1]
        num_func += pos[2] - 1
    total_num = num_loc + num_rel + num_func
    print('loc: ', num_loc, 'rel: ', num_rel, 'func: ', num_func, 'total: ', total_num)
    Pie(num_loc, num_rel, num_func, 'Groundtruth')


def Pie(num_loc, num_rel, num_func, fig_name=1):
    # pie figure
    plt.figure(fig_name)
    labels = ['LOCATE', 'RELATE', 'FUNC']
    sizes = [num_loc, num_rel, num_func]
    colors = [ 'yellowgreen', 'lightskyblue', 'yellow']
    patches, text1, text2 = plt.pie(sizes,
                                    labels=labels,
                                    colors=colors,
                                    autopct='%3.2f%%',
                                    shadow=False,
                                    startangle=90,
                                    pctdistance=0.6)
    plt.axis('equal')

    plt.title(fig_name)


if __name__ == '__main__':
    opt = parse_opt()
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    # print(vocab.word2idx)

    frame_features = h5py.File(opt.feature_h5_path, 'r')[opt.feature_h5_feats]
    h5 = h5py.File(opt.region_feature_h5_path, 'r')
    region_feats = h5[opt.region_visual_feats]
    spatial_feats = h5[opt.region_spatial_feats]

    # load pretrained model
    net = models.setup(opt, vocab)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    if not opt.eval_metric:
        net.load_state_dict(torch.load(opt.model_pth_path))
    elif opt.eval_metric == 'METEOR':
        net.load_state_dict(torch.load(opt.best_meteor_pth_path))
    elif opt.eval_metric == 'CIDEr':
        net.load_state_dict(torch.load(opt.best_cider_pth_path))
    else:
        raise ValueError('Please choose the metric from METEOR|CIDEr')
    net.to(DEVICE)
    net.eval()

    PrintStatistics('test')

    num_loc, num_rel, num_func = 0, 0, 0
    for vid in tqdm(range(*opt.test_range)):
        # print(vid)
        frame_feat = torch.from_numpy(frame_features[vid]).to(DEVICE).unsqueeze(0)
        region_feat = torch.from_numpy(region_feats[vid]).to(DEVICE).unsqueeze(0)
        spatial_feat = torch.from_numpy(spatial_feats[vid]).to(DEVICE).unsqueeze(0)
        words, loc_words, rel_words, func_words = sample(vocab, frame_feat, region_feat, spatial_feat, net, vid)
        num_loc += len(loc_words)
        num_rel += len(rel_words)
        num_func += len(func_words)

        with open(os.path.join(opt.result_dir, 'loc_words.txt'), 'a') as f:
            for word in loc_words:
                f.write(word + ' ')
            f.write('\n')

        with open(os.path.join(opt.result_dir, 'rel_words.txt'), 'a') as f:
            for word in rel_words:
                f.write(word + ' ')
            f.write('\n')

        with open(os.path.join(opt.result_dir, 'func_words.txt'), 'a') as f:
            for word in func_words:
                f.write(word + ' ')
            f.write('\n')


    total_num = num_loc + num_rel + num_func
    print('loc: ', num_loc, 'rel: ', num_rel, 'func: ', num_func, 'total: ', total_num)

    Pie(num_loc, num_rel, num_func, opt.result_dir)
    plt.show()