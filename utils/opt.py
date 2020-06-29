# coding: utf-8
import argparse
import time
import os

def parse_opt():
    # parser
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--dataset', type=str, default='msr-vtt', help='choose from msvd|msr-vtt')
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--save_per_epoch', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default=None, help='choose evaluate metric from METEOR|CIDEr|None')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--use_multi_gpu', action='store_true')

    # Network settings
    parser.add_argument('--model', type=str, default='RMN')
    parser.add_argument('--attention', type=str, default='gumbel', help='choose from gumbel|soft')
    parser.add_argument('--use_func', action='store_true', help='use func module or not')
    parser.add_argument('--use_rel', action='store_true', help='use relate module or not')
    parser.add_argument('--use_loc', action='store_true', help='use locate module or not')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--ss_factor', type=int, default=20)
    parser.add_argument('--frame_projected_size', type=int, default=1000)
    parser.add_argument('--region_projected_size', type=int, default=1000)
    parser.add_argument('--spatial_projected_size', type=int, default=300)
    parser.add_argument('--word_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=1300)
    parser.add_argument('--att_size', type=int, default=1024)
    parser.add_argument('--time_size', type=int, default=300)

    # Optimization settings
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--learning_rate_decay', type=int, default=1)
    parser.add_argument('--learning_rate_decay_every', type=int, default=10)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=10)
    parser.add_argument('--use_lin_loss', action='store_true', help='use linguistic loss or not')
    parser.add_argument('--lin_alpha', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=10)

    # Feature extract settings
    parser.add_argument('--max_frames', type=int, default=26)
    parser.add_argument('--max_words', type=int, default=26)
    parser.add_argument('--num_boxes', type=int, default=36)
    parser.add_argument('--a_feature_size', type=int, default=1536)
    parser.add_argument('--m_feature_size', type=int, default=1024)
    parser.add_argument('--region_feature_size', type=int, default=2048)
    parser.add_argument('--spatial_feature_size', type=int, default=5)
    parser.add_argument('--frame_shape', type=tuple, default=(3, 299, 299),
                        help='(3, 224, 224) for ResNet, (3, 299, 299) for InceptionV3 and InceptionResNetV2')
    parser.add_argument('--resnet_checkpoint', type=str, default='data/Backbone/resnet50-19c8e357.pth')
    parser.add_argument('--IRV2_checkpoint', type=str, default='data/Backbone/inceptionresnetv2-520b38e4.pth')
    parser.add_argument('--vgg_checkpoint', type=str, default='data/Backbone/vgg16-00b39a1b.pth' )
    parser.add_argument('--c3d_checkpoint', type=str, default='data/Backbone/c3d.pickle')

    # Dataset settings
    parser.add_argument('--msrvtt_video_root', type=str, default='data/MSR-VTT/Videos/')
    parser.add_argument('--msrvtt_anno_trainval_path', type=str, default='data/MSR-VTT/train_val_videodatainfo.json')
    parser.add_argument('--msrvtt_anno_test_path', type=str, default='data/MSR-VTT/test_videodatainfo.json')
    parser.add_argument('--msrvtt_anno_json_path', type=str, default='data/MSR-VTT/datainfo.json')
    parser.add_argument('--msrvtt_train_range', type=tuple, default=(0, 6513))
    parser.add_argument('--msrvtt_val_range', type=tuple, default=(6513, 7010))
    parser.add_argument('--msrvtt_test_range', type=tuple, default=(7010, 10000))

    parser.add_argument('--msvd_video_root', type=str, default='data/MSVD/youtube_videos')
    parser.add_argument('--msvd_csv_path', type=str, default='data/MSVD/video_corpus.csv')
    parser.add_argument('--msvd_video_name2id_map', type=str, default='data/MSVD/youtube_mapping.txt')
    parser.add_argument('--msvd_anno_json_path', type=str, default='data/MSVD/annotations.json')
    parser.add_argument('--msvd_train_range', type=tuple, default=(0, 1200))
    parser.add_argument('--msvd_val_range', type=tuple, default=(1200, 1300))
    parser.add_argument('--msvd_test_range', type=tuple, default=(1300, 1970))

    # Result path
    parser.add_argument('--result_dir', type=str, default='results/msr-vttgumbel')
    args = parser.parse_args()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    args.val_prediction_txt_path = os.path.join(args.result_dir, args.dataset + '_val_predictions.txt')
    args.val_score_txt_path = os.path.join(args.result_dir, args.dataset + '_val_scores.txt')
    args.test_prediction_txt_path = os.path.join(args.result_dir, args.dataset + '_test_predictions.txt')
    args.test_score_txt_path = os.path.join(args.result_dir, args.dataset + '_test_scores.txt')

    args.model_pth_path = os.path.join(args.result_dir, args.dataset + '_model.pth')
    args.best_meteor_pth_path = os.path.join(args.result_dir, args.dataset + '_best_meteor.pth')
    args.best_cider_pth_path = os.path.join(args.result_dir, args.dataset + '_best_cider.pth')
    args.optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_optimizer.pth')
    args.best_meteor_optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_best_meteor_optimizer.pth')
    args.best_cider_optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_best_cider_optimizer.pth')

    # caption and visual features path
    if args.dataset == 'msvd':
        args.feat_dir = 'data/MSVD'
    elif args.dataset == 'msr-vtt':
        args.feat_dir = 'data/MSR-VTT'
    else:
        raise ValueError('choose one dataset from msvd|msr-vtt')
    args.val_reference_txt_path = os.path.join(args.feat_dir, args.dataset + '_val_references.txt')
    args.test_reference_txt_path = os.path.join(args.feat_dir, args.dataset + '_test_references.txt')
    args.vocab_pkl_path = os.path.join(args.feat_dir, args.dataset + '_vocab.pkl')
    args.caption_pkl_path = os.path.join(args.feat_dir, args.dataset + '_captions.pkl')
    caption_pkl_base = os.path.join(args.feat_dir, args.dataset + '_captions')
    args.train_caption_pkl_path = caption_pkl_base + '_train.pkl'
    args.val_caption_pkl_path = caption_pkl_base + '_val.pkl'
    args.test_caption_pkl_path = caption_pkl_base + '_test.pkl'

    args.feature_h5_path = os.path.join(args.feat_dir, args.dataset + '_features.h5')
    args.feature_h5_feats = 'feats'
    args.feature_h5_lens = 'lens'

    if args.dataset == 'msvd':
        args.region_feature_h5_path = 'data/MSVD/msvd_region_feature.h5'
    elif args.dataset == 'msr-vtt':
        args.region_feature_h5_path = 'data/MSR-VTT/msrvtt_region_feature.h5'
    args.region_visual_feats = 'vfeats'
    args.region_spatial_feats = 'sfeats'

    args.video_sort_lambda = lambda x: int(x[5:-4])
    dataset = {
        'msr-vtt': [args.msrvtt_video_root, args.msrvtt_anno_json_path,
                    args.msrvtt_train_range, args.msrvtt_val_range, args.msrvtt_test_range],
        'msvd': [args.msvd_video_root, args.msvd_anno_json_path,
                 args.msvd_train_range, args.msvd_val_range, args.msvd_test_range]
    }
    args.video_root, args.anno_json_path, args.train_range, args.val_range, args.test_range = dataset[args.dataset]

    # tensorboard log path
    time_format = '%m-%d_%X'
    current_time = time.strftime(time_format, time.localtime())
    env_tag = '%s' % (current_time) + args.result_dir
    args.log_environment = os.path.join('logs', env_tag)

    return args


if __name__ == '__main__':
    opt = parse_opt()
    print(opt.feat_dir)
    print(opt.msrvtt_train_range)
    print(opt.video_root)
