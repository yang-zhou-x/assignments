(TODO)  
## 基于text CNN的中文新闻分类
本文使用 [**Keras**](https://keras.io/) 框架搭建对于中文新闻的文本分类模型。  
代码在[code.py](https://github.com/yang-zhou-x/assignments/blob/master/lstm_sentiment_analysis/code.py)  
对于文本的预处理包括标识化处理、移除停用词和标点符号、移除英文和数字、序列对齐等。  
部分处理流程参考了[Keras官方案例](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)。

运行过程：
![1554216327768.jpg]()

模型结构：
![1554216428391.jpg]()

## 数据来源

中文停用词、标点符号数据集由作者本人收集整理制作，共1663个，已上传至本页面。

## 环境配置
tensorflow-gpu==1.13.1  
keras-gpu==2.2.4  

## 作者
周扬，中国人民大学
