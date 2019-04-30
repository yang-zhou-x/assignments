# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG 
@Contact :   yzhou0000@gmail.com
'''

import os
import time
import numpy as np
from jieba import cut
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def time_elapse(func):
    """decorator, 计时。
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
def files_info(data_path, return_num=True):
    """返回文本类别标签，并进行分类计数，不读取文件。

    # Parameters
        data_path: str, 数据集的根目录
        return_num: (optinal) bool, 是否计数
    # Returns
        texts_info: dict[str:list], 标签类别（和各类文本数量）
    """
    texts_info = defaultdict(list)
    labels = os.listdir(data_path)
    for l in labels:
        texts_info['label'].append(l)
        if return_num:
            texts_path = os.path.join(data_path, l)
            texts_info['num'].append(len(os.listdir(texts_path)))
    return texts_info


@time_elapse
def get_texts_from_source(data_path):
    """读取文本。  

    # Parameters
        data_path: str, 数据集的根目录
    # Returns
        x_texts: list[str], 原始文本数据
        y_labels: list[str], 原始标签
    """
    labels_path = data_path
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


@time_elapse
def texts_tokenize(texts, stopwords=None, character_level=False):
    """获取分词后空格分隔的文本。

    # Parameters
        texts: list[str], 原始中文文本
        stopwords: (optional) set[str], 停用词/标点符号等
        character_level: (Optional) bool, 是否为单字级别
    # Returns
        texts: list[str], 分词后空格分隔的文本，去除了数字、英文(和停用词)
    """
    if stopwords is None:
        stopwords = ()
    for idx, t in tqdm(enumerate(texts), desc='Cutting texts'):
        res = (x for x in cut(t)
               if x not in stopwords and not x.encode('utf-8').isalnum())
        if character_level:
            texts[idx] = ' '.join(xx for x in res for xx in x)
        else:
            texts[idx] = ' '.join(res)
    return texts


def get_stopwords(word_path):
    """获取停用词/标点符号。

    # Parameters
        word_path: str, 停用词/标点符号表所在路径
    # Returns
        res: set[str], 停用词/标点符号
    """
    with open(word_path, encoding='utf-8') as f:
        res = f.readlines()  # list
    res = [x.strip('\n') for x in res] + ['\ufeff',
                                          ' ', '\t', '\n', '\r', '\u3000']
    return set(res)


@time_elapse
def texts_to_pad_sequences(x_train, x_test, dict_size, pad_len):
    """将分词文本转化为对齐后的整数序列。

    # Parameters
        x_train: list[str], 训练集
        x_test: list[str], 测试集
        dict_size: int, 字典大小（特征数量）
        pad_len: int, 对齐的长度
    # Returns
        x_train: list[str], 训练集
        x_test: list[str], 测试集
        tokenizer: Text tokenization utility class
    """
    tokenizer = Tokenizer(num_words=dict_size)
    tokenizer.fit_on_texts(x_train)
    x_train = pad_sequences(tokenizer.texts_to_sequences(x_train),
                            maxlen=pad_len, padding='pre', truncating='post')
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test),
                           maxlen=pad_len, padding='pre', truncating='post')
    return x_train, x_test, tokenizer


@time_elapse
def encode_y(y_labels, num_classes):
    """编码标签。

    # Parameters
        y_labels: list[str], 原始标签
        num_classes: int, 文本类别数量
    # Returns
        y_labels: array[int], one-hot编码后的标签
        le.classes_: list[str], 原始类别标签
    """
    le = LabelEncoder()
    y_labels = le.fit_transform(y_labels)
    if num_classes == 2:
        pass
    elif num_classes > 2:
        y_labels = to_categorical(y_labels, dtype='int8')
    else:
        raise ValueError('Wrong number of classes.')
    return y_labels, list(le.classes_)


def main():
    print('This module is used for pre-processing data.')
    print('Generally, it will take a long time to use get_texts_from_source() and texts_tokenize().')


if __name__ == '__main__':
    main()
