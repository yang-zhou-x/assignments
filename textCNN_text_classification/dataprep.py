# -*- coding: utf-8 -*-
'''
This module is used for transforming the raw data  
and dumping the reslut.

@Time    : 2019/04/13 17:55  
@Author  : ZHOU, YANG  
@Contact : yzhou0000@gmail.com
'''

import os
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import preprocess as pp


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

    Parameters
    ----------
        data_path: str, 数据集的根目录
        return_num: bool, (optinal) 是否计数
    Returns
    -------
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

    Parameters
    ----------
        data_path: str, 数据集的根目录
    Returns
    -------
        x_texts: list[str], 原始文本数据
        y_labels: list[str], 原始标签
    """
    labels_path = data_path
    labels = os.listdir(labels_path)
    tot = 0
    for l in labels:  # 遍历每个类别
        texts_path = os.path.join(labels_path, l)
        tot += len(os.listdir(texts_path))
    x_texts = [0] * tot
    y_labels = [0] * tot
    idx = 0
    for l in labels:
        texts_path = os.path.join(labels_path, l)
        texts_name = os.listdir(texts_path)
        for tn in tqdm(texts_name, desc=f'Reading {l}'):
            with open(os.path.join(texts_path, tn), encoding='utf-8') as f:
                x_texts[idx] = f.read().strip()
                y_labels[idx] = l
            idx += 1
    return x_texts, y_labels


def main():
    # Parameters
    use_stopwords = True
    character_level = False  # 是否为单字级别的token

    raw_set_name = 'THUCNews'
    stopwords_name = 'cn_stopwords_punctuations.csv'
    new_set_name = 'THUCNews.txt'
    cut_set_name = 'THUCNews_jieba.txt'

    # Paths
    init_path = os.getcwd()

    data_dir = os.path.join(init_path, 'datasets')
    stopwords_path = os.path.join(data_dir, stopwords_name)
    raw_set_path = os.path.join(data_dir, raw_set_name)
    new_set_path = os.path.join(data_dir, new_set_name)
    cut_set_path = os.path.join(data_dir, cut_set_name)

    # Transforming and Dumping
    x_texts, y_labels = get_texts_from_source(raw_set_path)
    pd.DataFrame({'label': y_labels, 'text': x_texts})\
        .to_csv(new_set_path, sep='\t', index=False, header=True)

    if use_stopwords:
        stopwords = pp.get_stopwords(stopwords_path)
        x_texts = pp.tokenize_texts(x_texts, stopwords, character_level)
    else:
        x_texts = pp.tokenize_texts(x_texts, character_level=character_level)

    pd.DataFrame({'label': y_labels, 'text': x_texts})\
        .to_csv(cut_set_path, sep='\t', index=False, header=True)


if __name__ == '__main__':
    main()
