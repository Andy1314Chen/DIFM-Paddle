## DIFM

### 一、简介

本项目是基于 PaddleRec 框架对 DIFM CTR 预估算法进行复现。

论文：[A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)

![DIFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffgzgk1bj30kq0e8wfz.jpg)

上图为 DIFM 的网络结构图，paper 题目中所指的 Dual-FEN 为 `vector-wise` 和 `bit-wise`两个 Input-aware Factorization 模块, 一个是 bit-wise,
一个是 vector-wise。只是维度上不同，实现的直觉是一样的。bit-wise 维度会对某一个 sparse embedding 向量内部彼此进行交叉，而 vector-wise 仅仅处理
embedding 向量层次交叉。把 vector-wise FEN 模块去掉，DIFM 就退化为 IFM 模型了，该算法也是论文作者实验组的大作，其结构图如下：

![IFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffi72287j60ez0cwq3p02.jpg)

两类不同维度的 FEN(Factor Estimating Net) 作用都是一致的，即输出 Embedding Layer 相应向量的权重。举个例子，假设上游有 n 个 sparse features， 
则 FEN 输出结果为 [a1, a2, ..., an]. 在 Reweighting Layer 中，对原始输入进行权重调整。最后输入到 FM 层进行特征交叉，输出预测结果。因此，总结两篇论文步骤如下：

- sparse features 经由 Embedding Layer 查表得到 embedding 向量，dense features 特征如何处理两篇论文都没提及；
- sparse features 对应的一阶权重也可以通过 1 维 Embedding Layer 查找；
- sparse embeddings 输入 FEN (bit-wise or vector-wise)，得到特征对应的权重 [a1, a2, ..., an]；
- Reweighting Layer 根据上一步骤中的特征权重，对 sparse embeddings 进一步调整；
- FM Layer 进行特征交叉，输出预测概率；


### 二、复现精度

本项目实现了 IFM、 DIFM 以及在 IFM 基础上增加了 deep layer 用于处理 dense features, 记作 IFM-Plus 的三种模型.
在 DIFM 论文中，两种算法在 Criteo 数据集的表现如下：

![](https://tva1.sinaimg.cn/large/008i3skNly1gtfg698y4nj30bo06tdgp.jpg)

本次 PaddlePaddle 论文复现赛要求在 PaddleRec Criteo 数据集上，DIFM 的复现精度为 AUC > 0.799. 

实际本项目复现精度为：
- IFM：AUC = 0.8016
- IFM-Plus: AUC = 0.8010
- DIFM: AUC = 0.799941

### 三、数据集

原论文采用 Kaggle Criteo 数据集，为常用的 CTR 预估任务基准数据集。单条样本包括 13 列 dense features、 26 列 sparse features及 label.

[Kaggle Criteo 数据集](https://www.kaggle.com/c/criteo-display-ad-challenge)
- train set: 4584, 0617 条
- test set:   604, 2135 条 （no label)

[PaddleRec Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
- train set: 4400, 0000 条
- test set:   184, 0617 条

P.S. 原论文所提及 Criteo 数据集为 Terabyte Criteo 数据集(即包含 1 亿条样本)，但作者并未使用全量数据，而是采样了连续 8 天数据进行训练和测试。
这个量级是和 PaddleRec Criteo 数据集是一样的，因此复现过程中直接选择了 PaddleRec 提供的数据。 原文表述如下：

![数据集介绍](https://tva1.sinaimg.cn/large/008i3skNly1gtgdgteholj61g40e6af502.jpg)


### 四、环境依赖
- 硬件：CPU、GPU
- 框架：
  - PaddlePaddle >= 2.1.2
  - Python >= 3.7

### 五、快速开始

该小节操作建议在百度 AI-Studio NoteBook 中进行执行。

AIStudio 项目链接：[https://aistudio.baidu.com/studio/project/partial/verify/2281174/3987013dd88e45ce828d3b9a3f2d24a9](https://aistudio.baidu.com/studio/project/partial/verify/2281174/3987013dd88e45ce828d3b9a3f2d24a9), 可以 fork 一下。

#### 1. AI-Studio 快速复现步骤
(约 6 个小时，可以直接在 notebook 切换版本加载预训练模型文件)

```
################# Step 1, git clone code ################
# 当前处于 /home/aistudio 目录, 代码存放在 /home/work/rank/DIFM-Paddle 中

import os
if not os.path.isdir('work/rank/DIFM-Paddle'):
    if not os.path.isdir('work/rank'):
        !mkdir work/rank
    !cd work/rank && git clone https://hub.fastgit.org/Andy1314Chen/DIFM-Paddle.git

################# Step 2, download data ################
# 当前处于 /home/aistudio 目录，数据存放在 /home/data/criteo 中

import os
os.makedirs('data/criteo', exist_ok=True)

# Download  data
if not os.path.exists('data/criteo/slot_test_data_full.tar.gz') or not os.path.exists('data/criteo/slot_train_data_full.tar.gz'):
    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_test_data_full.tar.gz
    
    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_train_data_full.tar.gz

################## Step 3, train model ##################
# 启动训练脚本 (需注意当前是否是 GPU 环境）
!cd work/rank/DIFM-Paddle && sh run.sh config_bigdata

```

#### 2. criteo slot_test_data_full 验证集结果
```
...
2021-08-14 11:53:10,026 - INFO - epoch: 0 done, auc: 0.799622, epoch time: 261.84 s
2021-08-14 11:57:32,841 - INFO - epoch: 1 done, auc: 0.799941, epoch time: 262.81 s
```

#### 3. 使用预训练模型进行预测
- 在 notebook 中切换到 `V1.2调参数` 版本，加载预训练模型文件，可快速验证测试集 AUC；
- ！！注意 config_bigdata.yaml 的 `use_gpu` 配置需要与当前运行环境保存一致 
```
!cd /home/aistudio/work/rank/DIFM-Paddle && python -u tools/infer.py -m models/rank/difm/config_bigdata.yaml
```

#### 4. 最优参数

```
  # 原文复现相关参数
  att_factor_dim: 80
  att_head_num: 16
  fen_layers_size:  [256, 256, 27]
  class: Adam
  learning_rate: 0.001
  train_batch_size: 2000
  epochs: 2
  
  # 简单调节 train_batch_size 到 1024，AUC 可以由 0.799941 提升到 0.801587
```
### 六、代码结构与详细说明

代码结构遵循 PaddleRec 框架结构
```
|--models
  |--rank
    |--difm                   # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类            
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # difm 核心算法代码，包括 difm 组网、ifm 组网等
|--tools                      # PaddleRec 工具类
|--LICENSE                    # 项目 LICENSE
|--README.md                  # readme
|--run.sh                     # 项目执行脚本(需在 aistudio notebook 中运行)
```

### 七、复现记录
1. 参考 PaddleRec 中 FM， 实现 IFM 模型，全量 Criteo 测试集上 AUC = 0.8016；
2. 在 IFM 模型基础上，增加 dnn layer 处理 dense features, 全量 Criteo 测试集上 AUC = 0.8010；
3. 在 IFM 模型基础上，增加 Multi-Head Self Attention，实现 DIFM；0.799941；
4. 增加 Multi-Head Self Attention 模块后，会导致模型显著过拟合，需要进一步细致调参，本项目参数直接参考论文默认参数，并未进行细粒度参数调优；