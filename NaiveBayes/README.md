## 基于PySpark的中文情感分析
本文使用 **PySpark** 框架搭建对于中文商品评论的分布式情感分析模型，在测试集上的准确率为85.48%。

模型基于 **TF-IDF** 和 **Naive Bayes** 构建。

对于文本的预处理包括标识化处理、移除停用词和标点符号等。

## 数据来源
商品评论数据集 https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
包含衣物、计算机、书籍、平板、水果等10个类别的6万余条评论数据，并且已标记好正向或负向。其中正向评论31728条，负向评论31046条，比例接近1:1，较为均衡。

中文停用词、标点符号数据集由作者本人收集整理制作，共1659个，已上传至本页面。

## 环境配置
Spark2.3.0

Python3.6.4
