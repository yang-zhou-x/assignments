## 基于LSTM的中文评论情感分析
本文使用[**Keras**](https://keras.io/)框架搭建对于中文商品评论的情感分析模型。训练2个epoch后在测试集上的准确率为90.42%。

数据预处理部分的代码在preprocess.py  
模型定义部分的代码在models.py  
模型训练部分的代码在train.py  
模型预测部分的代码在predict.py  
对于文本的预处理包括标识化处理、移除停用词和标点符号、移除英文和数字、序列对齐等。  
合并的代码在[code.py](https://github.com/yang-zhou-x/assignments/blob/master/lstm_sentiment_analysis/code.py)

运行过程：
![1554216327768.jpg](https://i.loli.net/2019/04/02/5ca37626ad6e1.jpg)

模型结构：
![1554216428391.jpg](https://i.loli.net/2019/04/02/5ca37626ac5c0.jpg)


**文本情感分析**：又被称为意见挖掘、倾向性分析、观点提取等，是指通过自然语言处理、文本挖掘方法等技术来识别和提取文本素材中所含的主观情绪信息。常见的应用包括给定一段文本，判断其所含有的是正面情绪还是负面情绪，本质上可以视作一个二分类问题。举例而言，商品评价“值得推荐!希望大家都读一下很有用的”是正向的，标签记为1；商品评价“像素低的很，还有破损”是负向的，标签记为0。  
情感分析的应用非常广泛，比较知名的有依靠社交网站Twitter的上公开信息进行情感分析以预测股市的走势，准确率可以达到87.6%，原文地址：[Twitter mood predicts the stock market](https://arxiv.org/pdf/1010.3003.pdf)

## 数据来源
商品评论数据集来自于 https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
。包含衣物、计算机、书籍、平板、水果等10个类别的6万余条评论数据，并且已标记好正向或负向。其中正向评论31728条，负向评论31046条，比例接近1:1，较为均衡。

中文停用词、标点符号数据集由作者本人收集整理制作，共1663个，已上传至本页面。

## 环境配置
CUDA==9.2  
cuDNN==7.3.1  
tensorflow-gpu==1.13.1  
Keras==2.2.4  

## 作者
周扬，中国人民大学
