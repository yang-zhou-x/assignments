# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG
@Contact :   yzhou0000@gmail.com
'''

import time
import pandas as pd
from tqdm import tqdm
from jieba import cut
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def time_elapse(desc=''):
    """decorator, 计时。

    # Parameters
        desc: str, 描述
    """
    def timeit(func):
        def aux(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            t1 = time.time()
            print('-' * 30)
            print(desc + f'used time: {round(t1-t0,2)s')
            print('-' * 30)
            return res
        return aux
    return timeit


@time_elapse
def reduce_memory_usage(df):
    """减小数据集占用的内存。
    通过合理降低数字精确度。不考虑时间型数据。

    # Parameters
        df: pandas.DataFrame
    # Returns
        df: pandas.DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024
    print(f'Memory usage: {round(start_mem, 2)} KB.')
    for col in df.columns:
        col_type = df[col].dtype  # 获取该列的数据类型
        if col_type != object:
            c_min = df[col].min()  # 数字大小的临界值
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # 对于整数型
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # 对于浮点型
                if c_min > np.finfo(np.float16).min \
                        and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min \
                        and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:  # 对于文本数据 或 类别型
            df[col] = df[col].astype('str')
            # df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024
    print(f'Memory usage: {round(end_mem, 2)} KB.')
    print(f'Reduced {round((start_mem - end_mem) / start_mem, 4) * 100}%.')
    return df


@time_elapse
def get_data_from_csv(data_path):
    """从csv文档读取数据，并打印数据集信息。

    # Parameters
        data_path: str, 数据集路径
    # Returns
        df: pd.DataFrame, 数据集
    """
    df = pd.read_csv(data_path, header=0)
    print(f'the shape of data is {df.shape}.')
    try:
        df['review'] = df['review'].astype('str')
        print('计算每个类别评论长度的中位数、平均值、标准差：')
        df['review_len'] = df['review'].map(lambda x: len(x))
        print(df[['cat', 'review_len']]
              .groupby(by='cat')
              .agg(['median', 'mean', 'std']))
    except:
        pass
    return df


@time_elapse
def get_stopwords(stopwords_path):
    """获取停用词表。

    # Parameters
        stopwords_path: str, 停用词表路径
    # Returns
        words: set[str], 停用词
    """
    with open(stopwords_path, encoding='utf-8') as f:
        words = f.readlines()
    words = set(x.strip() for x in words)
    others = {'\ufeff', ' '}
    words = words | others
    return words


@time_elapse
def tokenize_texts(texts, stopwords=None, character_level=False):
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


@time_elapse
def texts_to_pad_sequences(x_train, pad_len, dict_size=None,
                           x_test=None, tokenizer=None):
    """将分词文本转化为对齐后的整数序列。

    # Parameters
        x_train: list[str], 训练集
        pad_len: int, 对齐的长度
        dict_size: (optional) int, 字典大小（特征数量）
        x_test: (optional) list[str], 测试集
        tokenizer: (optional) keras text tokenization utility class
    # Returns
        x_train: list[str], 训练集
        x_test: (optional) list[str], 测试集
        tokenizer: keras text tokenization utility class
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=dict_size)
        tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=pad_len,
                            padding='pre', truncating='post')
    if x_test is not None:
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=pad_len,
                               padding='pre', truncating='post')
        return x_train, x_test, tokenizer
    return x_train, tokenizer


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


if __name__ = '__main__':
    main()
