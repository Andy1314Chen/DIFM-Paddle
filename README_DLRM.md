#### 一、基于 PaddleRec 框架的 DLRM 推荐算法复现

##### 1. AI-Studio 快速复现步骤
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

##### 2. criteo slot_test_data_full 验证集结果
```
2021-08-09 14:08:14,539 - INFO - read data
2021-08-09 14:08:14,539 - INFO - reader path:criteo_reader
2021-08-09 14:08:14,540 - INFO - load model epoch 0
2021-08-09 14:08:14,540 - INFO - start load model from output_model_dlrm/0
2021-08-09 14:08:15,721 - INFO - epoch: 0, batch_id: 0, auc: 0.856177,accuracy: 0.000000, avg_reader_cost: 0.00187 sec, avg_batch_cost: 0.00201 sec, avg_samples: 256.00000, ips: 55497.64 ins/s
...
2021-08-09 14:13:46,552 - INFO - epoch: 0, batch_id: 6656, auc: 0.804113,accuracy: 0.000000, avg_reader_cost: 0.01508 sec, avg_batch_cost: 0.05010 sec, avg_samples: 256.00000, ips: 5107.50 ins/s
2021-08-09 14:13:58,920 - INFO - epoch: 0, batch_id: 6912, auc: 0.804086,accuracy: 0.000000, avg_reader_cost: 0.01553 sec, avg_batch_cost: 0.04829 sec, avg_samples: 256.00000, ips: 5298.94 ins/s
2021-08-09 14:14:11,539 - INFO - epoch: 0, batch_id: 7168, auc: 0.804185,accuracy: 0.000000, avg_reader_cost: 0.01239 sec, avg_batch_cost: 0.04927 sec, avg_samples: 256.00000, ips: 5193.93 ins/s
2021-08-09 14:14:12,513 - INFO - epoch: 0 done, auc: 0.804220,accuracy: 0.000000, epoch time: 357.97 s
```

==2021-08-09 14:14:12,513 - INFO - epoch: 0 done, auc: 0.804220,accuracy: 0.000000, epoch time: 357.97 s==，
达到要求的 AUC>0.79, 复现成功！

##### 3. 利用训练好的模型文件快速验证
- 复现 DLRM 保存了训练好的模型文件，链接: https://pan.baidu.com/s/1EXnl9KlzTRehuxlQ70lUCQ  密码: msr1
- 解压后放在 tools 同级目录下，再利用以下命令可以快速验证测试集 AUC：
```
!cd /home/aistudio/work/rank/DLRM-Paddle && python -u tools/infer.py -m models/rank/dlrm/config_bigdata.yaml
```



#### 二、DLRM 算法原理

![DLRM](https://tva1.sinaimg.cn/large/008i3skNly1gt8kwo40g9j30ei0cmjru.jpg)

1. 模型结构

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

2. Embedding

待补充...


3. Experiments

大佬发文章就是 NB，DLRM vs DCN without extensive tuning and no regularization is used. 简简单单的 SGD + lr=0.1
就把 Accuracy 干上去了。。。

![实验结果](https://tva1.sinaimg.cn/large/008i3skNly1gta7vj34mkj30ty0c8abt.jpg)


#### 三、复现记录
1. 2021-08-04 tensorflow2 版本 
- pytorch 实现代码翻译为 tensorflow2 版本， 在 criteo 测试集上取得 accuracy >= 0.80 & auc >= 0.79

2. 2021-08-06 基于 PaddleRec 框架版本
- 参考 tensorflow2 实现代码，基于 PaddleRec 框架实现 DLRM，在 sample data 上成功运行

3. 2021-08-07 paddle 版本跑全量 criteo 数据集 & ~~调参~~ 炼丹！
- batch_size: 128
- Ai-Studio CPU 跑的太慢了... 申请一下 GPU 资源

4. 2021-08-08 Ai-Studio GPU 点卡用完了 & 周末无法申请 GPU 资源
- CPU 上运行，增大 batch_size & 增大学习率，减少 epoch
- 核心参数 {epochs: 2, batch_size: 2048, optimizer: SGD, learning_rate: 0.1}
- slot_test_data_full 全量测试集上 AUC = 0.804146


#### 四、遇到问题
1. 训练结束进行验证集测试时，会遇到 "EOFError: marshal data too short" 报错，可能要清理一下 __pycache__ 文件
2. GPU 资源太少了。。。



#### 五、参考资料
1. [PaddleRec 文档](README_CN.md)
2. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091v1.pdf)
3. [Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
4. [DLRM Pytorch 实现](https://github.com/facebookresearch/dlrm)

