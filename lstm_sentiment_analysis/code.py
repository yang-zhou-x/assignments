# -*- coding: utf-8 -*-
"""
Created on 2019
@author: Yang Zhou
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from jieba import cut

from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 设置模型参数
max_features = 10000  # 词汇表的单词总数
embedding_dim = 128  # 词向量的维度
max_len = 30  # 序列的长度
output_dim = 64  # LSTM的输出维度
drop_out_rate = 0.2
batch_size = 64
optimizer = Adam(lr=.002, epsilon=.001)
epoch = 5
metric = ['accuracy']

'''
1.数据准备
'''
print('准备数据...')
init_path = os.getcwd()
# 读取商品评论数据
data = pd.read_csv(filepath_or_buffer=os.path.join(init_path, 'datasets/online_shopping_10_cats.csv'), header=0)
print(f'原始数据集的形状为: {data.shape}')

# 每个类别的评论长度
data['review'] = data['review'].astype('str')
print('计算每个类别评论长度的中位数、平均值、标准差：')
data['len'] = data['review'].map(lambda x: len(x))
print(data[['cat', 'len']].groupby(by='cat').agg([np.median, np.mean, np.std]))


# 修饰器，计时
def time_func(desc=''):
    def timeit(func):
        def aux(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            t1 = time.time()
            print(desc + f'耗时{round(t1 - t0, 3)}s.')
            return res
        return aux
    return timeit


# 减少数据集的内存占用
@time_func('减少数据集内存占用')
def reduce_memory_usage(df):
    """
    通过合理降低数字精确度，减少数据集对于内存的占用。
    不考虑时间型数据。
    # 参数
        df: pandas.DataFrame
    # return
        pandas.DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024
    print(f'数据集初始占用内存大小为： {round(start_mem, 2)} KB.')
    for col in df.columns:
        col_type = df[col].dtype  # 获取该列的数据类型
        if col_type != object:
            c_min = df[col].min()  # 数字大小的临界值
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # 对于整数型
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # 对于浮点型
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:  # 对于文本数据 或 类别型
            df[col] = df[col].astype('str')
            # df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024
    print(f'数据集处理后占用内存大小： {round(end_mem, 2)} KB.')
    print(f'减少了{round((start_mem - end_mem) / start_mem, 4) * 100}%.')
    return df


data = reduce_memory_usage(data)


# 获取停用词、标点符号
@time_func('获取停用词/标点符号')
def get_stop_words():
    """
    读取停用词表，含标点符号
    # return
        list[str], 停用词表/符号
    """
    def iter_file(path):
        with open(path, encoding='utf-8') as file:
            for line in file:
                yield line
    res = []
    for word in iter_file(os.path.join(init_path, 'datasets/cn_stopwords_punctuations.csv')):
        res.append(word.strip('\n'))
    res += ['\ufeff', ' ']
    return res


stops = get_stop_words()


# 分词
def pre_transform(string, stop_words):
    """
    对一条评论进行分词、去除停用词和标点符号后，以空格分隔重新拼接
    # return
        str, 分好词的评论
    """
    return ' '.join(x for x in cut(string) if x not in stop_words)


print('进行分词...')
t = time.time()
data.loc[:, 'review'] = data['review'].map(lambda x: pre_transform(x, stops))
print(f'分词耗时{round(time.time() - t, 3)}s.')

# 将文本转化为整数序列
print('将文本转化为整数序列...')
X = data['review'].values
token = text.Tokenizer(num_words=max_features)
token.fit_on_texts(X)
X = token.texts_to_sequences(X)

# 将序列截断或补齐为相同长度
print('将序列截断或补齐为相同长度...')
X = sequence.pad_sequences(X, maxlen=max_len, padding='pre', truncating='post')

# 划分训练集和测试集
print('划分训练集和测试集...')
x_train, x_test, y_train, y_test = train_test_split(X, data['label'].values,
                                                    test_size=0.3, random_state=2019)
print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')

'''
2.建立模型
'''
print('建立模型...')
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len))
# 使用全部序列(return_sequences=True)
model.add(LSTM(units=output_dim, recurrent_activation='hard_sigmoid', return_sequences=True,
               recurrent_dropout=drop_out_rate))
model.add(Activation('tanh'))
model.add(Dropout(drop_out_rate))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

print('模型概况：')
model.summary()

# 模型编译
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=metric)


'''
3.训练模型并评价
'''
print('训练模型并评价...')
callback = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          callbacks=callback,
          validation_data=(x_test, y_test))


'''
4.保存模型 model persistence
'''
save_dir = os.path.join(init_path, 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'lstm_online_shopping.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'模型被保存在{model_path}')
