#!/usr/bin/env zsh

# command examples
# MSVD
python train.py --dataset=msvd --model=RMN \
 --result_dir=results/msvd_model --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=10 \
 --learning_rate_decay_rate=10 \
 --use_loc --use_rel --use_func \
 --learning_rate=1e-4 --attention=gumbel \
 --hidden_size=512 --att_size=512 \
 --train_batch_size=64 --test_batch_size=32

python evaluate.py --dataset=msvd --model=RMN \
 --result_dir=results/msvd_model --attention=gumbel \
 --use_loc --use_rel --use_func \
 --hidden_size=512 --att_size=512 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr

python sample.py --dataset=msvd --model=RMN \
 --result_dir=results/msvd_model --attention=gumbel \
 --use_loc --use_rel --use_func \
 --hidden_size=512 --att_size=512 \
 --eval_metric=CIDEr

# MSR-VTT
python train.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=5 \
 --learning_rate_decay_rate=3 \
 --use_loc --use_rel --use_func --use_multi_gpu \
 --learning_rate=1e-4 --attention=gumbel \
 --hidden_size=1300 --att_size=1024 \
 --train_batch_size=32 --test_batch_size=8

python evaluate.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr

python sample.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --eval_metric=CIDEr
