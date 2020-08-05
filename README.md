# Learning to Discretely Compose Reasoning Module Networks for Video Captioning (IJCAI2020)
## Introduction
In this [paper](https://www.ijcai.org/Proceedings/2020/0104.pdf), we propose a novel visual reasoning approach for video captioning, 
named Reasoning Module Networks (RMN), to equip the existing encoder-decoder 
framework with reasoning capacity. Specifically, our RMN employs 1) 
three sophisticated spatio-temporal reasoning modules, 
and 2) a dynamic and discrete module selector trained by a linguistic loss with
a Gumbel approximation. This code is the Pytorch implementation of our work.
![image](https://github.com/tgc1997/RMN/blob/master/models/framework.png)


## Dependencies
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
2. Download visual and text features of [MSVD](https://rec.ustc.edu.cn/share/f9335ba0-ba07-11ea-9198-9366c81a1928) 
and [MSR-VTT](https://rec.ustc.edu.cn/share/26685ac0-ba08-11ea-866f-6fc664dfaa3b), and put them in `data` folder.
3. Download pre-trained models [msvd_model](https://rec.ustc.edu.cn/share/711b44e0-ba08-11ea-848d-b3f63a452975) 
and [msr-vtt_model](https://rec.ustc.edu.cn/share/84993310-ba08-11ea-8055-0f1d6ef31a0d), and put them in `results` folder.

> Download instruction ([#1](https://github.com/tgc1997/RMN/issues/1)): 1. enter the folder, 2. choose all files, 3. download.


## Evaluation
We provide the pre-trained models of "RMN(H+L)" in the paper to reproduce the result reported in paper. 
Note that because the MSVD dataset is too small, the training result is not stable, so the final result of MSVD in
the paper is the average of three training results.

Metrics | MSVD | MSR-VTT
:-: | :-: | :-: 
BLEU@4 | 56.4 | 42.5 |
METEOR | 37.2 | 28.4 |
ROUGE-L| 74.0 | 61.6 |
CIDEr  | 97.8 | 49.6 |

Evaluation command example:
```python
python evaluate.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --test_batch_size=2 --beam_size=2 \
 --eval_metric=CIDEr
```

## Training
You can also train you own model by running
Training command example:
```python
python train.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=5 \
 --learning_rate_decay_rate=3 \
 --use_loc --use_rel --use_func --use_multi_gpu \
 --learning_rate=1e-4 --attention=gumbel \
 --hidden_size=1300 --att_size=1024 \
 --train_batch_size=32 --test_batch_size=8
```
You can also add `--use_multi_gpu` to train the model with multiply GPUs.

## Sampleing 
Sampleing command example:
```python
python sample.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msr-vtt_model \
 --use_loc --use_rel --use_func \
 --hidden_size=1300 --att_size=1024 \
 --eval_metric=CIDEr
```
By running this command, you can get the pie chart in the paper. And when uncommenting the 
visualization code in `sample.py`, you can visualize the module selection process.

## Video Captioning Papers
[This repository](https://github.com/tgc1997/Awesome-Video-Captioning) contains a curated list of research papers in Video Captioning(from 2015 to 2020). Link to the code and project website if available.

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{tan2020learning,
title={Learning to Discretely Compose Reasoning Module Networks for Video Captioning},
author={Tan, Ganchao and Liu, Daqing and Wang Meng and Zha, Zheng-Jun},
booktitle={IJCAI-PRICAI},
year={2020}
}
```
