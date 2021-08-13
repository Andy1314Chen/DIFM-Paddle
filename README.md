## DIFM

### 一、简介

本项目是基于 PaddleRec 框架对 DIFM CTR 预估算法进行复现。

论文：[A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)

![DIFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffgzgk1bj30kq0e8wfz.jpg)

上图为 DIFM 的网络结构图，paper 题目中所指的 Dual-FEN 即是 `vector-wise` 和 `bit-wise`两个 Input-aware Factorization 模块, 一个是 bit-wise,
一个是 vector-wise, 实现的直觉是一样的，只是维度上不同。bit-wise 维度会对某一个 sparse embedding 向量内部彼此进行交叉，而 vector-wise 仅仅处理
embedding 向量维度交叉。把 vector-wise FEN 去掉，就退化为 IFM 模型了，该模型也是论文作者实验组的大作，其结构图如下：

![IFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffi72287j60ez0cwq3p02.jpg)

两类不同维度的 FEN(Factor Estimating Net) 其实结果都是输出对 Embedding Layer 相应向量的权重，假设上游有 n 个 sparse features， 则 FEN 输出结果
为 [a1, a2, ..., an]. 在 Reweighting Layer 中，对原始输入进行权重调整。最后输入到 FM 层进行特征交叉，输出预测结果。因此，总结两篇论文步骤如下：

- sparse features 经由 Embedding Layer 查表得到 embedding 向量，dense features 特征如何处理两篇论文都没提及；
- sparse features 对应的一阶权重也可以通过 1 维 Embedding Layer 查找；
- sparse embeddings 输入 FEN (bit-wise or vector-wise)，得到特征对应的权重 [a1, a2, ..., an]；
- Reweighting Layer 根据特征权重，对 sparse embeddings 进一步调整；
- FM Layer 进行特征交叉，输出预测概率


### 二、复现精度

本项目实现了 IFM 和 DIFM，在 IFM 基础上增加了 deep layer 用于处理 dense features, 记作 IFM-plus. 在 DIFM 论文中，两种算法在 Criteo 数据集的表现如下：

![](https://tva1.sinaimg.cn/large/008i3skNly1gtfg698y4nj30bo06tdgp.jpg)

本次 PaddlePaddle 论文复现赛要求在 Kaggle Criteo 数据集上，DIFM 的复现精度为 AUC > 0.799. 

实际本项目复现精度为：
- IFM：AUC = 0.8016
- IFM-plus: AUC = 0.8010 (测试集每个 epoch 均超过 0.8) 
- DIFM: AUC = 

### 三、数据集

原论文采用 Kaggle Criteo 数据集，为常用的 CTR 预估任务基准数据集。单条样本包括 13 列 dense features、 26 列 sparse features及 label.

[Kaggle Criteo 数据集](https://www.kaggle.com/c/criteo-display-ad-challenge)
- train set: 4584, 0617 条
- test set:   604, 2135 条 （no label)

[PaddleRec Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
- train set: 4400, 0000 条
- test set:   184, 0617 条

本项目采用 PaddleRec 所提供的 Criteo 数据集进行复现。

### 四、环境依赖
- 硬件：CPU、GPU
- 框架：
  - PaddlePaddle >= 2.1.2
  - Python >= 3.7

### 五、快速开始

该小节操作建议在百度 AI-Studio NoteBook 中进行执行。

AIStudio 项目链接：[x](x), 可以 fork 一下。

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
2021-08-11 18:19:45,528 - INFO - epoch: 0, batch_id: 5888, auc: 0.805084,accuracy: 0.793505, avg_reader_cost: 0.02961 sec, avg_batch_cost: 0.05567 sec, avg_samples: 256.00000, ips: 4596.73 ins/s
2021-08-11 18:19:59,916 - INFO - epoch: 0, batch_id: 6144, auc: 0.805157,accuracy: 0.793632, avg_reader_cost: 0.03085 sec, avg_batch_cost: 0.05618 sec, avg_samples: 256.00000, ips: 4554.94 ins/s
2021-08-11 18:20:14,480 - INFO - epoch: 0, batch_id: 6400, auc: 0.805081,accuracy: 0.793623, avg_reader_cost: 0.02785 sec, avg_batch_cost: 0.05687 sec, avg_samples: 256.00000, ips: 4499.95 ins/s
2021-08-11 18:20:30,772 - INFO - epoch: 0, batch_id: 6656, auc: 0.805203,accuracy: 0.793568, avg_reader_cost: 0.02980 sec, avg_batch_cost: 0.06361 sec, avg_samples: 256.00000, ips: 4023.01 ins/s
2021-08-11 18:20:46,270 - INFO - epoch: 0, batch_id: 6912, auc: 0.805174,accuracy: 0.793536, avg_reader_cost: 0.02354 sec, avg_batch_cost: 0.06051 sec, avg_samples: 256.00000, ips: 4228.88 ins/s
2021-08-11 18:21:00,821 - INFO - epoch: 0, batch_id: 7168, auc: 0.805253,accuracy: 0.793609, avg_reader_cost: 0.02986 sec, avg_batch_cost: 0.05682 sec, avg_samples: 256.00000, ips: 4504.05 ins/s
2021-08-11 18:21:01,991 - INFO - epoch: 0 done, auc: 0.805245,accuracy: 0.793599, epoch time: 424.70 s
```

#### 3. 使用预训练模型进行预测
- 在 notebook 中切换到 V1.0 版本，加载预训练模型文件，可快速验证测试集 AUC；
- ！！注意 config_bigdata.yaml 的 `use_gpu` 配置需要与当前运行环境保存一致 
```
!cd /home/aistudio/work/rank/DIFM-Paddle && python -u tools/infer.py -m models/rank/difm/config_bigdata.yaml
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
