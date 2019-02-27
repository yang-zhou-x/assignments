# -*- coding: utf-8 -*-
"""
Created on 2018-12-29
@author: Yang Zhou, zhouyang0995@ruc.edu.cn
"""

import numpy as np
import pandas as pd
import jieba
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

'''
3.1.初始化并准备数据
'''
# 初始化
sc = SparkContext('local', appName='comment_clf')  # Master: local

# 读入HDFS上的文件
lines = sc.textFile('~/online_shopping_10_cats.csv')

# 去掉首行
header = lines.first()
lines = lines.filter(lambda row: row != header)
# 查看前2行
lines.take(2)

parts = lines.map(lambda row: row.split(','))
parts.take(2)

# 模型所需数据
labels = parts.map(lambda ele: float(ele[1]))
labels.take(2)
comments = parts.map(lambda ele: ele[2])
comments.take(2)

'''
3.2.标识化处理
'''
# 标识化
comments_tokenized = comments.map(lambda ele: jieba.lcut(ele.strip()))
comments_tokenized.take(2)

'''
3.3.移除停用词和标点
'''
# 停用词和标点符号，本地
stop_words = pd.read_csv('～/cn_stop_punctuations.csv')
stop_words = np.array(stop_words)
stop_words = stop_words.reshape(1659)
stop_words = stop_words.tolist()
print(stop_words[:10])

stops = stop_words + ['\ufeff']  # 非法字符'\ufeff'
print(stops[1590:])

# 清理标点符号、停用词、非法字符等
comments_clean = comments_tokenized.map(lambda ele: [e for e in ele if e not in stops])
comments_clean.take(2)

# 定义features数量
hashingTF = HashingTF(5000)

'''
3.4.TF-IDF
'''
# tf-idf
comments_tf = hashingTF.transform(comments_clean)
comments_idf = IDF().fit(comments_tf)
comments_tfidf = comments_idf.transform(comments_tf)

'''
3.5.朴素贝叶斯
'''
# 合并RDD
final_data = labels.zip(comments_tfidf)

# 划分训练集和测试集
train_set, test_set = final_data.randomSplit([0.8, 0.2], seed=20182019)
train_rdd = train_set.map(lambda ele: LabeledPoint(ele[0], ele[1]))
test_rdd = test_set.map(lambda ele: LabeledPoint(ele[0], ele[1]))

# 训练
clf_nb = NaiveBayes.train(train_rdd)

# 预测
nb_output = test_rdd.map(lambda p: (clf_nb.predict(p.features), p.label))

# 前5个预测结果
nb_output.take(5)

# 准确率
nb_accuracy = nb_output.map(lambda ele: ele[0] == ele[1]).mean()
print(nb_accuracy)

# 退出上下文环境
sc.stop()
