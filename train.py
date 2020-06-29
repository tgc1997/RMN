import shutil
import pickle
import time
from utils.utils import *
from utils.data import get_train_loader
from utils.opt import parse_opt

import models
import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
from tensorboard_logger import configure, log_value

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(opt):
    configure(opt.log_environment, flush_secs=10)

    # load vocabulary
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(vocab_size)

    # print parameters
    print('Learning rate: %.5f' % opt.learning_rate)
    print('Learning rate decay: ', opt.learning_rate_decay)
    print('Batch size: %d' % opt.train_batch_size)
    print('results directory: ', opt.result_dir)

    # build model
    net = models.setup(opt, vocab)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    print('Total parameters:', sum(param.numel() for param in net.parameters()))

    if os.path.exists(opt.model_pth_path) and opt.use_checkpoint:
        net.load_state_dict(torch.load(opt.model_pth_path))
    net.to(DEVICE)

    # initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
    if os.path.exists(opt.optimizer_pth_path) and opt.use_checkpoint:
        optimizer.load_state_dict(torch.load(opt.optimizer_pth_path))

    # initialize data loader
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path,
                                    opt.train_batch_size)
    total_step = len(train_loader)

    # prepare groundtruth
    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)

    # start training
    best_meteor = 0
    best_meteor_epoch = 0
    best_cider = 0
    best_cider_epoch = 0
    loss_count = 0
    count = 0
    saving_schedule = [int(x * total_step / opt.save_per_epoch) for x in list(range(1, opt.save_per_epoch + 1))]
    print('total: ', total_step)
    print('saving_schedule: ', saving_schedule)
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        if opt.learning_rate_decay and epoch % opt.learning_rate_decay_every == 0 and epoch > 0:
            opt.learning_rate /= opt.learning_rate_decay_rate
        epsilon = max(0.6, opt.ss_factor / (opt.ss_factor + np.exp(epoch / opt.ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        log_value('epsilon', epsilon, epoch)

        for i, (frames, regions, spatials, captions, pos_tags, cap_lens, video_ids) in enumerate(train_loader, start=1):
            # convert data to DEVICE mode
            frames = frames.to(DEVICE)
            regions = regions.to(DEVICE)
            spatials = spatials.to(DEVICE)
            pos_tags = pos_tags.to(DEVICE)
            targets = captions.to(DEVICE)

            # compute results of the model
            optimizer.zero_grad()
            outputs, module_weights = net(frames, regions, spatials, targets, epsilon)
            tokens = outputs
            bsz = len(captions)

            # remove pad and flatten outputs
            outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
            outputs = outputs.view(-1, vocab_size)

            # remove pad and flatten targets
            targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
            targets = targets.view(-1)

            # compute captioning loss
            cap_loss = criterion(outputs, targets)

            # compute linguistic loss
            lin_loss = torch.tensor(0).to(DEVICE)
            if opt.use_loc and opt.use_rel and opt.use_func:
                # remove pad and flatten module weights
                module_weights = torch.cat([module_weights[j][:cap_lens[j]] for j in range(bsz)], 0)
                module_weights = module_weights.view(-1, 3)

                # remove pad and flatten pos_tags
                pos_tags = torch.cat([pos_tags[j][:cap_lens[j]] for j in range(bsz)], 0)
                pos_tags = pos_tags.view(-1)

                # compute linguistic loss
                lin_loss = criterion(module_weights, pos_tags)

            # compute total loss
            if opt.use_lin_loss:
                total_loss = cap_loss + opt.lin_alpha * lin_loss
            else:
                total_loss = cap_loss

            log_value('cap_loss', cap_loss.item(), epoch * total_step + i)
            log_value('lin_loss', lin_loss.item(), epoch * total_step + i)
            log_value('total_loss', total_loss.item(), epoch * total_step + i)
            loss_count += total_loss.item()
            total_loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            if i % 10 == 0 or bsz < opt.train_batch_size:
                loss_count /= 10 if bsz == opt.train_batch_size else i % 10
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                      (epoch, opt.max_epoch, i, total_step, loss_count,
                       np.exp(loss_count)))
                loss_count = 0
                tokens = tokens.max(2)[1]
                tokens = tokens.data[0].squeeze()
                if opt.use_multi_gpu:
                    we = net.module.decoder.decode_tokens(tokens)
                    gt = net.module.decoder.decode_tokens(captions[0].squeeze())
                else:
                    we = net.decoder.decode_tokens(tokens)
                    gt = net.decoder.decode_tokens(captions[0].squeeze())
                print('[vid:%d]' % video_ids[0])
                print('WE: %s\nGT: %s' % (we, gt))

            if i in saving_schedule:
                torch.save(net.state_dict(), opt.model_pth_path)
                torch.save(optimizer.state_dict(), opt.optimizer_pth_path)

                blockPrint()
                start_time_eval = time.time()
                net.eval()

                # use opt.val_range to find the best hyperparameters
                metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference)
                end_time_eval = time.time()
                enablePrint()
                print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))

                for k, v in metrics.items():
                    log_value(k, v, epoch * len(saving_schedule) + count)
                    print('%s: %.6f' % (k, v))
                    if k == 'METEOR' and v > best_meteor:
                        shutil.copy2(opt.model_pth_path, opt.best_meteor_pth_path)
                        shutil.copy2(opt.optimizer_pth_path, opt.best_meteor_optimizer_pth_path)
                        best_meteor = v
                        best_meteor_epoch = epoch

                    if k == 'CIDEr' and v > best_cider:
                        shutil.copy2(opt.model_pth_path, opt.best_cider_pth_path)
                        shutil.copy2(opt.optimizer_pth_path, opt.best_cider_optimizer_pth_path)
                        best_cider = v
                        best_cider_epoch = epoch

                print('Step: %d, Learning rate: %.8f' % (epoch * len(saving_schedule) + count, opt.learning_rate))
                optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
                log_value('Learning rate', opt.learning_rate, epoch * len(saving_schedule) + count)
                count += 1
                count %= 4
                net.train()

        end_time = time.time()
        print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))

    with open(opt.test_score_txt_path, 'w') as f:
        f.write('MODEL: {}\n'.format(opt.model))
        f.write('best meteor epoch: {}\n'.format(best_meteor_epoch))
        f.write('best cider epoch: {}\n'.format(best_cider_epoch))
        f.write('Learning rate: {:6f}\n'.format(opt.learning_rate))
        f.write('Learning rate decay: {}\n'.format(opt.learning_rate_decay))
        f.write('Batch size: {}\n'.format(opt.train_batch_size))
        f.write('results directory: {}\n'.format(opt.result_dir))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)