## 基于PySpark的中文情感分析
本文使用[**PySpark**](https://spark.apache.org/docs/latest/api/python/index.html)框架搭建对于中文商品评论的分布式情感分析模型，在测试集上的准确率为85.48%。  
模型基于 **TF-IDF** 和 **Naive Bayes** 构建。代码在[code.py](https://github.com/yang-zhou-x/assignments/blob/master/naiveBayes_sentiment_analysis/code.py)  
对于文本的预处理包括标识化处理、移除停用词和标点符号等。

移除停用词/标点符号后的分词结果（部分）：
![1.png](https://i.loli.net/2019/03/02/5c7a759638c13.png)

在测试集上的准确率：
![2.png](https://i.loli.net/2019/03/02/5c7a7596009d1.png)


**文本情感分析**：又被称为意见挖掘、倾向性分析、观点提取等，是指通过自然语言处理、文本挖掘方法等技术来识别和提取文本素材中所含的主观情绪信息。常见的应用包括给定一段文本，判断其所含有的是正面情绪还是负面情绪，本质上可以视作一个二分类问题。举例而言，商品评价“值得推荐!希望大家都读一下很有用的”是正向的，标签记为1；商品评价“像素低的很，还有破损”是负向的，标签记为0。

情感分析的应用非常广泛，比较知名的有依靠社交网站Twitter的上公开信息进行情感分析以预测股市的走势，准确率可以达到87.6%，原文地址：https://arxiv.org/pdf/1010.3003.pdf *Twitter mood predicts the stock market.*

## 数据来源
商品评论数据集来自于 https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
。包含衣物、计算机、书籍、平板、水果等10个类别的6万余条评论数据，并且已标记好正向或负向。其中正向评论31728条，负向评论31046条，比例接近1:1，较为均衡。

中文停用词、标点符号数据集由作者本人收集整理制作，共1659个，已上传至本页面。

## 环境配置
CentOS 6.10  
Spark 2.3.0  
Python 3.6.4

## 作者
周扬，中国人民大学
