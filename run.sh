#!/usr/bin/env zsh

# command examples
python train.py --dataset=msvd --model=RMN \
 --result_dir=results/test --use_lin_loss \
 --use_loc --use_rel --use_func \
 --learning_rate=1e-4 --attention=gumbel \
 --hidden_size=512 --att_size=512 \
 --train_batch_size=16 --test_batch_size=16

python evaluate.py --dataset=msvd --model=RMN \
 --result_dir=results/msvdgumbel3 --attention=gumbel \
 --use_loc --use_rel --use_func \
 --hidden_size=512 --att_size=512 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr

python sample.py --dataset=msvd --model=RMN \
 --result_dir=results/msvdgumbel3 --attention=gumbel \
 --use_loc --use_rel --use_func \
 --hidden_size=512 --att_size=512 \
 --eval_metric=CIDEr

python evaluate.py --dataset=msr-vtt --model=RMN \
 --result_dir=results \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr

python sample.py --dataset=msr-vtt --model=RMN \
 --result_dir=results \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --eval_metric=CIDEr
