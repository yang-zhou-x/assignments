
## 基于梯度提升树(GBDT)的玩家排名预测
使用LightGBM对《绝地求生：大逃杀》玩家的最终百分比排名进行预测。  
代码包含两大部分：（1）特征工程；（2）模型。详细内容及说明见代码注释。

**LightGBM**是一个实现GBDT算法的框架，由微软DMTK(分布式机器学习工具包)团队在GitHub上开源，具有以下优点：  
(1)更快的训练速度  
(2)更低的内存消耗  
(3)更好的准确率  
(4)分布式支持，可以快速处理海量数据  
与基于预排序（pre-sorted）决策树算法的GBDT工具相比，LightGBM使用基于直方图（histogram）的算法。在分割增益的复杂度方面，histogram算法只需要计算O(#bins)次, 远少于pre-sorted算法的O(#data)，并且可以通过直方图的相减来进行进一步的加速。

原始特征的相关图：  
![图片 1.png](https://i.loli.net/2019/03/02/5c7a70b3c429b.png)

不同赛制的玩家数量：  
![图片 2.png](https://i.loli.net/2019/03/02/5c7a70f3ad01a.png)

特征的重要性排名：  
![图片 3.png](https://i.loli.net/2019/03/02/5c7a70f3c203d.png)

## 数据来源
约445万条记录，包含近30个特征。  
来自于PUBG的官方公开数据 https://www.kaggle.com/c/pubg-finish-placement-prediction

## 环境配置
macOS 10.13.6  
gcc 8.2.0  
cmake 3.13.4  
libomp 7.0.0  
lightgbm 2.2.3

## 作者
周扬，中国人民大学
