# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG 
@Contact :   yzhou0000@gmail.com
'''

import os
import time
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import text, sequence
from keras.utils import to_categorical


def time_elapse(func):
    """
    decorator, 计时
    """
    def aux(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        print('-' * 30)
        print(f'Used time: {round(t1-t0,2)} seconds.')
        print('-' * 30)
        return res
    return aux


@time_elapse
def files_info(init_path, return_num=False):
    """
    返回文本类别标签，并进行分类计数，不读取文件。
    
    # 参数
        init_path: str, 项目的根目录
        return_num: (optinal) bool, 是否计数
    # return
        texts_info: dict[str:list], 标签类别，（和各类文本数量）
    """
    texts_info = defaultdict(list)
    labels_path = os.path.join(init_path, 'datasets/THUCNews')
    labels = os.listdir(labels_path)
    for l in labels:
        texts_info['label'].append(l)
        if return_num:
            texts_path = os.path.join(labels_path, l)
            texts_info['num'].append(len(os.listdir(texts_path)))
    return texts_info


@time_elapse
def get_texts(init_path):
    """
    读取文本。  
    
    # 参数
        init_path: str, 项目的根目录
    # return
        x_texts: list[str], 原始文本数据
        y_labels: list[str], 原始标签
    """
    labels_path = os.path.join(init_path, 'datasets/THUCNews')
    labels = os.listdir(labels_path)
    tot = 0
    for l in labels:
        texts_path = os.path.join(labels_path, l)
        tot += len(os.listdir(texts_path))
    x_texts = [0] * tot
    y_labels = [0] * tot
    idx = 0
    for l in labels:
        texts_path = os.path.join(labels_path, l)
        texts_name = os.listdir(texts_path)
        for tn in tqdm(texts_name, desc=f'读取{l}类'):
            with open(os.path.join(texts_path, tn), encoding='utf-8') as f:
                x_texts[idx] = f.read().strip()
                y_labels[idx] = l
            idx += 1
    return x_texts, y_labels
    # Used Memory: about 1604MB


@time_elapse
def text_tokenize(x_texts, stopwords=None, use_stopwords=False):
    """
    文本分词。
    
    # 参数
        x_texts: list[str], 原始文本数据
        stopwords: (optional) set[str], 停用词/标点符号等
        use_stopwords: (optional) bool, 是否用停用词。否，则过滤掉长度小于2的词。
    # return
        x_texts: list[str], 分词后的文本数据
    """
    def tokenize(string):
        if use_stopwords:
            return ' '.join(x for x in jieba.cut(string)
                            if x not in stopwords and not x.encode('utf-8').isalnum())
        else:
            return ' '.join(x for x in jieba.cut(string)
                            if len(x) > 1 and not x.encode('utf-8').isalnum())
    x_texts = map(tokenize, x_texts)
    return list(x_texts)


def get_stopwords(word_path):
    """
    获取停用词/标点符号。
    
    # 参数
        word_path: str, 停用词/标点符号表所在路径
    # return
        res: set[str], 停用词/标点符号
    """
    with open(word_path, encoding='utf-8') as f:
        res = f.readlines()
    res = [x.strip('\n') for x in res] + ['\ufeff', ' ']
    return set(res)


@time_elapse
def texts_to_pad_sequences(x_train, x_test, dict_size, pad_len):
    """
    将分词文本转化为对齐后的整数序列。
    
    # 参数
        x_train: list[str], 训练集
        x_test: list[str], 测试集
        dict_size: int, 字典大小（特征数量）
        pad_len: int, 对齐的长度
    # return
        x_train: list[str], 训练集
        x_test: list[str], 测试集
        token.index_word: 词索引
    """
    token = text.Tokenizer(num_words=dict_size)
    token.fit_on_texts(x_train)
    x_train = sequence.pad_sequences(token.texts_to_sequences(x_train),
                                     maxlen=pad_len, padding='pre', truncating='post')
    x_test = sequence.pad_sequences(token.texts_to_sequences(x_test),
                                    maxlen=pad_len, padding='pre', truncating='post')
    return x_train, x_test, token.index_word
    # Used time: 437.38 seconds.


@time_elapse
def encode_y(y_labels):
    """
    编码标签。
    
    # 参数
        y_labels: list[str], 原始标签
    # return
        y_labels: array[int], one-hot编码后的标签
        le.classes_: list[str], 文本标签
    """
    le = LabelEncoder()
    y_labels = le.fit_transform(y_labels)
    y_labels = to_categorical(y_labels, dtype='int32')
    return y_labels, list(le.classes_)


def main():
    print('Generally, it will take a long time to use get_texts() and text_tokenize().')


if __name__ == '__main__':
    main()
