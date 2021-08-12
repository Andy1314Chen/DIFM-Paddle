<<<<<<< HEAD
=======
## DLRM

### 一、简介

本项目是基于 PaddleRec 框架对 2019 年 Facebook 提出的 DLRM CTR 排序算法进行复现。

论文：[Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091v1.pdf)


![DLRM](https://tva1.sinaimg.cn/large/008i3skNly1gt8kwo40g9j30ei0cmjru.jpg)


推荐 rank 模型一般较为简单，如上图 DLRM 的网络结构看着和 DNN 就没啥区别，主要由四个基础模块构成，`Embeddings`、 `Matrix Factorization`、`Factorization Machine`和`Multilayer Perceptrons`。

DLRM 模型的特征输入，主要包括 dense 数值型和 sparse 类别型两种特征。dense features 直接连接 MLP（如图中的蓝色三角形），
sparse features 经由 embedding 层查找得到相应的 embedding 向量。Interactions 层进行特征交叉（包含 dense features 和 sparse features 的交叉及
sparse features之间的交叉等），与因子分解机 FM 有些类似。

DLRM 模型中所有的 sparse features 的 embedding 向量长度均是相等的，且dense features 经由 MLP 也转化成相同的维度。这点是理解该模型代码的关键。

- dense features 经过 MLP (bottom-MLP) 处理为同样维度的向量
- spare features 经由 lookup 获得统一维度的 embedding 向量（可选择每一特征对应的 embedding 是否经过 MLP 处理）
- dense features & sparse features 的向量两两之间进行 dot product 交叉
- 交叉结果再和 dense 向量 concat 一起输入到顶层 MLP (top-MLP)  
- 经过 sigmoid 函数激活得到点击概率


### 二、复现精度

原论文意在介绍 DLRM 的网络结构，对模型参数并未进行细致调节，与 baseline DCN 算法对比实验结果中如下所示：

![实验结果](https://tva1.sinaimg.cn/large/008i3skNly1gta7vj34mkj30ty0c8abt.jpg)

在 Kaggle Criteo 数据集上，不同梯度更新方法结果不同，复现精度 AUC > 0.79 & Accuracy > 0.79.

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

AIStudio 项目链接：[https://aistudio.baidu.com/aistudio/projectdetail/2263714](https://aistudio.baidu.com/aistudio/projectdetail/2263714), 可以 fork 一下。

#### 1. AI-Studio 快速复现步骤

```
################# Step 1, git clone code ################
# 当前处于 /home/aistudio 目录, 代码存放在 /home/work/rank/DLRM-Paddle 中

import os
if not os.path.isdir('work/rank/DLRM-Paddle'):
    if not os.path.isdir('work/rank'):
        !mkdir work/rank
    !cd work/rank && git clone https://hub.fastgit.org/Andy1314Chen/DLRM-Paddle.git

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
!cd work/rank/DLRM-Paddle && sh run.sh config_bigdata

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
- 复现 DLRM 保存了训练好的模型文件，链接: https://pan.baidu.com/s/1EXnl9KlzTRehuxlQ70lUCQ  密码: msr1
- 解压后放在 tools 同级目录下，再利用以下命令可以快速验证测试集 AUC：
```
!cd /home/aistudio/work/rank/DLRM-Paddle && python -u tools/infer.py -m models/rank/dlrm/config_bigdata.yaml
```

### 六、代码结构与详细说明

代码结构遵循 PaddleRec 框架结构
```
|--models
  |--rank
    |--dlrm                   # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类            
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # dlrm 核心算法代码，包括 dlrm 组网等
|--tools                      # PaddleRec 工具类
|--LICENSE                    # 项目 LICENSE
|--README.md                  # readme
|--README-old.md              # 原始 readme
|--README_CN.md               # PaddleRec 中文 readme
|--README_EN.md               # PaddleRec 英文 readme
|--run.sh                     # 项目执行脚本(需在 aistudio notebook 中运行)
```

>>>>>>> main
