# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG 
@Contact :   yzhou0000@gmail.com
'''

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, Input, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import getData_THUCNews


# 设定参数
num_target_classes = 14  # 文本类别数量
dict_size = 10000  # 词典词数
max_sequence_len = 256  # 文本序列的长度
embedding_dimension = 128  # 词嵌入维度
num_filters = 2  # 卷积核数量
pool_size = 3  # 池化层核宽
drop_out_rate = .2
hidden_dim = 64  # 倒数第2个全连接层
optimizer = Adam(lr=.0005)
batch_size = 128  # 用于训练和测试
epochs = 50  # 实际epoch<=50
metrics = ['accuracy']
init_path = os.getcwd()
data_path = os.path.join(init_path, 'datasets/THUCNews')  # 数据集路径
save_dir = os.path.join(init_path, 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'textCNN_THUCNews.hdf5'  # 模型名称
model_path = os.path.join(save_dir, model_name)  # 模型持久化路径


def _get_last_layer_options(num_classes):
    """获得最后一个Dense层的units数量、激活函数，以及损失函数。

    # 参数
        num_classes: int, 输出类别的数量。
    # Returns
        units数量: int
        激活函数: str
        损失函数: str
    """
    if num_classes == 2:
        units = 1
        activation = 'sigmoid'
        loss_func = 'binary_crossentropy'
    elif num_classes > 2:
        units = num_classes
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    else:
        raise ValueError('Wrong Number of Classes')
    return [units, activation, loss_func]


def text_cnn_model(num_features,
                   sequence_len,
                   embedding_dim,
                   filters,
                   pool_size_,
                   dropout_rate,
                   use_pretrained_embedding=False,
                   is_embedding_trainable=False,
                   embedding_matrix=None):
    """创建一个text CNN模型的实例。

    # 参数
        num_features: int, 词典字数，也是embedding的输入维度。
        sequence_len: int, 序列的长度。
        embedding_dim: int, embedding的输出维度。
        filters: int, 滤波器数量，每层的输出维度（channel数量）。
        dropout_rate: float, Dropout层对于输入的drop比例。
        use_pretrained_embedding: bool, 是否使用预训练的embedding。
        is_embedding_trainable: bool, true if embedding layer is trainable.
        embedding_matrix: dict, dictionary with embedding coefficients.
    # Returns
        一个text CNN模型实例。
    """
    inputs = Input(shape=(max_sequence_len,))
    # 词嵌入层。可以添加预训练的词向量。
    if use_pretrained_embedding:
        embed = Embedding(input_dim=num_features,
                          output_dim=embedding_dim,
                          input_length=sequence_len,
                          weights=[embedding_matrix],
                          trainable=is_embedding_trainable)(inputs)
    else:
        embed = Embedding(input_dim=num_features,
                          output_dim=embedding_dim,
                          input_length=sequence_len)(inputs)
    # 卷积层、池化层
    cnns = []
    for size in (2, 3, 4):
        cnn = Conv1D(filters=filters, kernel_size=size,
                     padding='same', strides=1)(embed)
        cnn = Activation('relu')(cnn)
        cnn = MaxPooling1D(pool_size=pool_size_, padding='valid')(cnn)
        cnns.append(cnn)
    merge = concatenate(cnns, axis=1)
    # Drop out
    outputs = Dropout(dropout_rate)(merge)
    outputs = Flatten()(outputs)
    # 全连接层
    outputs = Dense(hidden_dim)(outputs)
    outputs = Activation('relu')(outputs)
    last_units, last_activation = options_last_layer[:2]
    outputs = Dense(last_units)(outputs)
    outputs = Activation(last_activation)(outputs)
    model_ = Model(inputs=inputs, outputs=outputs)
    return model_


def main():
    print('Loading data...')
    x_texts, y_labels = getData_THUCNews.get_texts(data_path)
    
    stopword_path = os.path.join(
        init_path, 'datasets/cn_stopwords_punctuations.csv')
    stopwords = getData_THUCNews.get_stopwords(stopword_path)
    
    x_texts = getData_THUCNews.text_tokenize(x_texts, stopwords, True)
    y_labels, labels_name = getData_THUCNews.encode_y(y_labels, num_target_classes)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_texts, y_labels, test_size=0.2, random_state=2019)
    x_train, x_test, index_word = getData_THUCNews.texts_to_pad_sequences(
        x_train, x_test, dict_size, max_sequence_len)
    
    # 确定模型的最后一层
    options_last_layer = _get_last_layer_options(num_target_classes)

    # 创建一个模型实例
    print('Building model...')
    model = text_cnn_model(sequence_len=max_sequence_len,
                           num_features=dict_size,
                           embedding_dim=embedding_dimension,
                           filters=num_filters,
                           pool_size_=pool_size,
                           dropout_rate=drop_out_rate)
    model.summary()

    # 模型编译
    model.compile(optimizer=optimizer,
                  loss=options_last_layer[2],
                  metrics=metrics)
    # 回调
    callback = [EarlyStopping(monitor='val_acc', patience=5),
                ModelCheckpoint(model_path, monitor='val_acc',
                                verbose=1, save_best_only=True)]

    # 模型训练
    print('Training model...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback,
                        validation_data=(x_test, y_test))

    # acc和loss的可视化
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

    plot_acc_loss()

    # 载入保存好的最佳模型
    best_model = load_model(model_path)

    # 测试
    y_pred = best_model.predict(x_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 输出结果：precision,recall,F1
    clf_report = classification_report(
        y_true, y_pred, target_names=labels_name)
    print(clf_report)

    # 绘制混淆矩阵热力图
    conf_m = confusion_matrix(y_true, y_pred)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax = sns.heatmap(conf_m, xticklabels=labels_name, yticklabels=labels_name,
                     annot=True, fmt="d", cmap="YlGnBu")
    plt.show()


if __name__ == '__main__':
    main()
