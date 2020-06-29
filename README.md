# Learning to Discretely Compose Reasoning Module Networks for Video Captioning (IJCAI2020)
## Introduction
In this [paper](), we propose a novel visual reasoning approach for video captioning, 
named Reasoning Module Networks (RMN), to equip the existing encoder-decoder 
framework with reasoning capacity. Specifically, our RMN employs 1) 
three sophisticated spatio-temporal reasoning modules, 
and 2) a dynamic and discrete module selector trained by a linguistic loss with
a Gumbel approximation. This code is the Pytorch implementation of our work.
![image](https://github.com/tgc1997/RMN/blob/master/models/framework.png)


## Environment
* Python 3.7 (other versions may also work)
* Pytorch 1.1.0 (other versions may also work)
* pickle
* tqdm
* h5py
* matplotlib
* numpy
* tensorboard_logger

## Prepare
1. Create two empty folders, `data` and `results`
2. Download [visual and text features]() of MSVD and MSR-VTT, and put them in `data` folder.
3. Download [pre-trained models](), and put them in `results` folder.


## Evaluation
We provide the pre-trained models of "RMN(H+L)" in the paper to reproduce the result reported in paper. 
Note that because the MSVD dataset is too small, the training result is not stable, so the result of MSVD in
the paper is the average of three training results.

Metrics | MSVD | MSR-VTT
:-: | :-: | :-: 
BLEU@4 | 54.6 | 42.5 |
METEOR | 36.5 | 28.4 |
ROUGE-L| 73.4 | 61.6 |
CIDEr  | 94.4 | 49.6 |

Evaluation command example:
```python
python evaluate.py --dataset=msr-vtt --model=RMN \
 --result_dir=results \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr
```

## Training
Training command example:
```python
python train.py --dataset=msvd --model=RMN \
 --result_dir=results/test --use_lin_loss \
 --use_loc --use_rel --use_func \
 --learning_rate=1e-4 --attention=gumbel \
 --hidden_size=512 --att_size=512 \
 --train_batch_size=16 --test_batch_size=16
```
You can also add `--use_multi_gpu` to train the model with multiply GPUs.

## Sampleing 
Sampleing command example:
```python
python sample.py --dataset=msvd --model=RMN \
 --result_dir=results/msvdgumbel3 --attention=gumbel \
 --use_loc --use_rel --use_func \
 --hidden_size=512 --att_size=512 \
 --eval_metric=CIDEr
```
By running this command, you can get the pie chart in the paper. And when uncommenting the 
visualization code in `sample.py`, you can visualize the module selection process.

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@article{
}
```
