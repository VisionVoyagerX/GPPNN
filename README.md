# PanFormer

PanFormer pansharpenning method implemented in pytorch

Pretrained model is provided

Based on implementation: https://github.com/shuangxu96/GPPNN

Paper link: https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Deep_Gradient_Projection_Networks_for_Pan-sharpening_CVPR_2021_paper.pdf

# Dataset

The GaoFen-2 and WorldView-3 dataset download links can be found in https://github.com/liangjiandeng/PanCollection

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GPPNN                                    [1, 4, 256, 256]          --
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-1                      [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-1               [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-2               [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-3               [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-2                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-4               [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-5               [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-6               [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-3                      [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-7               [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-8               [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-9               [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-4                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-10              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-11              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-12              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-5                      [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-13              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-14              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-15              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-6                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-16              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-17              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-18              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-7                      [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-19              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-20              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-21              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-8                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-22              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-23              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-24              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-9                      [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-25              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-26              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-27              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-10                    [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-28              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-29              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-30              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-11                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-31              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-32              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-33              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-12                    [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-34              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-35              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-36              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-13                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-37              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-38              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-39              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-14                    [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-40              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-41              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-42              [1, 4, 256, 256]          512
├─ModuleList: 1-15                       --                        (recursive)
│    └─LRBlock: 2-15                     [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-43              [1, 4, 256, 256]          4,608
│    │    └─BasicUnit: 3-44              [1, 4, 64, 64]            4,608
│    │    └─BasicUnit: 3-45              [1, 4, 256, 256]          4,608
├─ModuleList: 1-16                       --                        (recursive)
│    └─PANBlock: 2-16                    [1, 4, 256, 256]          --
│    │    └─BasicUnit: 3-46              [1, 1, 256, 256]          320
│    │    └─BasicUnit: 3-47              [1, 4, 256, 256]          320
│    │    └─BasicUnit: 3-48              [1, 4, 256, 256]          512
==========================================================================================
Total params: 119,808
Trainable params: 119,808
Non-trainable params: 0
Total mult-adds (G): 5.59
==========================================================================================
Input size (MB): 0.33
Forward/backward pass size (MB): 1431.31
Params size (MB): 0.48
Estimated Total Size (MB): 1432.11
==========================================================================================

```

# Quantitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Figures.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Figures.png?raw=true)

# Qualitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Images.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Images.png?raw=true)
