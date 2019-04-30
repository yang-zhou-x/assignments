# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG 
@Contact :   yzhou0000@gmail.com
'''

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import preprocess as pp
import models as m

# 设定参数
seed = 2019
num_target_classes = 14  # 文本类别数量
dict_size = 10000  # 词汇表大小
character_level = False  # 是否为单字级别的token
use_stopwords = True
max_sequence_len = 256  # 序列对齐的长度
embedding_dimension = 128  # 词嵌入维度
num_filters = 2  # 卷积核数量
pool_size = 3  # 池化层核宽
drop_out_rate = .2
fc_units = 64  # 输出层前的全连接层的units数量
optimizer = Adam(lr=.0005)
batch_size = 128  # 用于训练和测试
epochs = 50  # 实际epoch<=50
metrics = ['accuracy']
init_path = os.getcwd()
data_path = os.path.join(init_path, 'datasets/THUCNews')  # 数据集路径
stopwords_path = os.path.join(
    init_path, 'datasets/cn_stopwords_punctuations.csv')  # 停用词路径
save_dir = os.path.join(init_path, 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'TextCNN_THUCNews.hdf5'  # 模型名称
model_path = os.path.join(save_dir, model_name)  # 模型持久化路径
tokenizer_path = os.path.join(save_dir, 'tokenizer.pickle')  # tokenizer路径


def main():
    print('-' * 30)
    print('Loading data...')
    x_texts, y_labels = pp.get_texts_from_source(data_path)

    print('-' * 30)
    print('Computing the mean/max/median/min length of texts...')
    data_raw = pd.DataFrame({'label': y_labels, 'text': x_texts})
    data_raw['len_text'] = data_raw['text'].map(lambda x: len(x))
    print(data_raw[['label', 'len_text']].groupby(by='label', sort=False)
          .agg(['mean', 'max', 'median', 'min']))

    print('-' * 30)
    print('Tokenizing...')
    if use_stopwords:
        stopwords = pp.get_stopwords(stopwords_path)
    else:
        stopwords = None
    x_texts = pp.texts_tokenize(x_texts, stopwords, character_level)

    print('-' * 30)
    print('Encoding y...')
    y_labels, labels_name = pp.encode_y(
        y_labels, num_target_classes)

    print('-' * 30)
    print('Spliting train/test datasets...')
    x_train, x_test, y_train, y_test = train_test_split(
        x_texts, y_labels, test_size=0.2, random_state=seed)

    print('-' * 30)
    print('Vectorizing texts and padding sequences...')
    x_train, x_test, tokenizer = pp.texts_to_pad_sequences(
        x_train, x_test, dict_size, max_sequence_len)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)  # preserve tokenizer

    # 确定模型的最后一层
    options_last_layer = m.get_last_layer_options(num_target_classes)

    print('-' * 30)
    print('Building model...')
    model = m.text_cnn_model(num_features=dict_size,
                             sequence_len=max_sequence_len,
                             embedding_dim=embedding_dimension,
                             filters=num_filters,
                             pool_size=pool_size,
                             dropout_rate=drop_out_rate,
                             fc_units=fc_units,
                             options_last_layer=options_last_layer)
    model.summary()

    print('-' * 30)
    print('Compiling model...')
    model.compile(optimizer=optimizer,
                  loss=options_last_layer[2],
                  metrics=metrics)
    # 回调
    callback = [EarlyStopping(monitor='val_acc', patience=5),
                ModelCheckpoint(model_path, monitor='val_acc',
                                verbose=1, save_best_only=True)]

    print('-' * 30)
    print('Training model...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback,
                        validation_data=(x_test, y_test))

    def get_max_val_acc_idx():
        max_val_acc = max(history.history['val_acc'])
        max_idx = history.history['val_acc'].index(max_val_acc) + 1
        print('-' * 30)
        print(f'The max val_acc is {max_val_acc}, when epoch is {max_idx}.')

    get_max_val_acc_idx()

    print('-' * 30)
    print('Plotting the acc and loss...')

    def plot_acc_loss():
        history_dict = history.history
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epoch = range(1, len(acc) + 1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epoch, loss, label='Training loss')
        plt.plot(epoch, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epoch, acc, label='Training acc')
        plt.plot(epoch, val_acc, label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    plt.rcParams['font.sans-serif'] = ['Source Han Sans TW']
    plot_acc_loss()

    print('-' * 30)
    print('Loading the best model...')
    best_model = load_model(model_path)

    print('-' * 30)
    print('Predicting...')
    y_pred = best_model.predict(x_test,
                                batch_size=batch_size,
                                verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 输出结果：precision,recall,F1
    print('-' * 30)
    print('Printing classification report...')
    clf_report = classification_report(y_true, y_pred,
                                       target_names=labels_name,
                                       digits=4)
    print(clf_report)

    # 绘制混淆矩阵热力图
    print('-' * 30)
    print('Plotting confusion matrix...')
    conf_m = confusion_matrix(y_true, y_pred)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ax = sns.heatmap(conf_m, xticklabels=labels_name, yticklabels=labels_name,
                     annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Confusion Matrix on Predictions')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
