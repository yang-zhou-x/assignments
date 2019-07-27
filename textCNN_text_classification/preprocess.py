# -*- coding: utf-8 -*-
'''
This module is used for pre-processing data.

@Time    : 2019
@Author  : ZHOU, YANG  
@Contact : yzhou0000@gmail.com  
'''

import jieba
import pkuseg
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline


def get_stopwords(word_path):
    """获取停用词。

    Parameters
    ----------
        word_path: str, 停用词表所在路径
    Returns
    -------
        words: set[str], 停用词
    """
    with open(word_path, encoding='utf-8') as f:
        words = f.readlines()  # list
    words = set(x.strip() for x in words)
    others = {'\ufeff', ' ', '\t', '\n', '\r', '\u3000'}
    words = words | others
    return words


def tokenize_texts(texts,
                   stopwords=None,
                   character_level=False,
                   tool='jieba'):
    """Tokenization. 获取分词后空格分隔的句子。

    Parameters
    ----------
        texts: list[str], 原始中文文本
        stopwords: set[str], (optional) 可使用停用词
        character_level: bool, (optional) 是否为单字级别
        tool: str, (optional) 分词使用'jieba'或'pkuseg'
    Returns
    -------
        texts: list[str], 分词后空格分隔的句子, in-place
    """
    if tool == 'jieba':
        tokenizer = jieba.cut
    elif tool == 'pkuseg':
        seg = pkuseg.pkuseg(model_name='default')
        tokenizer = seg.cut
    else:
        raise ValueError("The value of parameter `tool` should be \
            'jieba' or 'pkuseg'.")
    if stopwords is None:
        stopwords = set()
    for idx, t in tqdm(enumerate(texts), desc='Cutting texts'):
        res = (x for x in tokenizer(t) if x not in stopwords)
        if character_level:
            texts[idx] = ' '.join(xx for x in res for xx in x)
        else:
            texts[idx] = ' '.join(res)
    return texts


def texts_to_sequence_vectors(x_train, pad_len,
                              dict_size=None,
                              x_test=None,
                              tokenizer=None):
    """Vectorization. 将已分词文本转换为sequences向量。包括：  
    (1)将每条文本转换为整数序列。序列中每个数字代表该词在  
    词典中的索引。索引数字依据频数大小。  
    (2)序列对齐。

    `tokenizer`可以从零训练，也可以使用已保存的。使用已保存的`tokenizer`
    时，`x_train`和`x_test`无需区分。

    Parameters
    ----------
        x_train: list[str], 训练集，分词后空格分隔
        pad_len: int, 序列对齐的长度
        dict_size: int, (optional) 字典大小（即特征数量）
        x_test: list[str], (optional) 测试集，分词后空格分隔
        tokenizer: (optional) keras text tokenization utility class
    Returns
    -------
        x_train: list[str], 训练集文本向量
        x_test: list[str], (optional) 测试集文本向量
        tokenizer: Text tokenization utility class
    """
    if tokenizer is None:
        if dict_size is None:
            raise ValueError('If `tokenizer` is None, \
                `dict_size` must be specified.')
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


def texts_to_ngram_vectors(train_texts,
                           test_texts=None,
                           ngram_range=(1, 1),
                           use_tfidf=True):
    """Vectorization. 将已分词文本转换为N-gram向量。

    Parameters
    ----------
        train_texts: list[str], 训练集，分词后空格分隔
        test_texts: list[str], (optional) 测试集，分词后空格分隔
        ngram_range: (int,int), (optional) N-gram中N的取值范围
        use_tfidf: bool, (optional) 是否使用TF-IDF
    Returns
    -------
        train_texts: list[str], 训练集文本向量
        test_texts: list[str], (optional) 测试集文本向量
        index_word: list[str], 整数索引到字词的mapping
    """
    count = CountVectorizer(token_pattern=r'(?u)\b\w+\b',
                            ngram_range=ngram_range)
    train_texts = count.fit_transform(train_texts)
    if use_tfidf:
        tfidf = TfidfTransformer()
        train_texts = tfidf.fit_transform(train_texts)
        pipe = make_pipeline(count, tfidf)
    if test_texts is not None:
        try:
            test_texts = pipe.transform(test_texts)
        except NameError:
            test_texts = count.transform(test_texts)
    index_word = count.get_feature_names()
    return train_texts, test_texts, index_word


def encode_y(y_labels, num_classes):
    """编码标签。

    Parameters
    ----------
        y_labels: list[str], 原始标签
        num_classes: int, 文本类别数量
    Returns
    -------
        y_labels: np.ndarray[int], one-hot编码后的标签
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


def reduce_memory_usage(df):
    """减小数据集占用的内存。  
    通过合理降低数字精确度。不考虑时间型数据。

    Parameters
    ----------
        df: pandas.DataFrame
    Returns
    -------
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


def main():
    print('This module is used for pre-processing data.')


if __name__ == '__main__':
    main()
