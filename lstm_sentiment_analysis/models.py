# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG
@Contact :   yzhou0000@gmail.com
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Embedding, LSTM


def get_last_layer_options(num_classes):
    """获取输出层的units数量、激活函数、损失函数。

    # Parameters
        num_classes: int, 输出类别的数量。
    # Returns
    tuple, including:
        units: int, Dense层的units数量
        activation: str, 激活函数
        loss_func: str, 损失函数
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
        raise ValueError('Wrong Number of Classes.')
    return (units, activation, loss_func)


def lstm_model(num_features,
               embedding_dim,
               sequence_len,
               lstm_units,
               dropout_rate,
               last_layer_options,
               use_pretrained_embedding=False,
               embedding_matrix=None,
               is_embedding_trainable=False):
    """创建一个LSTM模型的实例。

    # Parameters
        num_features: int, 词汇表大小，也是embedding层的输入维度
        embedding_dim: int, 词向量维度，即embedding层的输出维度
        sequence_len: int, 输入序列的长度
        lstm_units: int, LSTM层的unit数量
        dropout_rate: float, Dropout层对于输入的drop比例
        last_layer_options: tuple, 输出层的units数量、激活函数、损失函数
        use_pretrained_embedding: bool, 是否使用预训练的embedding
        is_embedding_trainable: bool, true if embedding layer is trainable
        embedding_matrix: dict, dictionary with embedding coefficients
    # Returns
        model: 一个LSTM模型实例
    """
    model = Sequential()
    # 词嵌入层。可以添加预训练的词向量。
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=sequence_len,
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=sequence_len))
    model.add(LSTM(units=lstm_units,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   dropout=0.0,
                   recurrent_dropout=0.0,
                   implementation=1,
                   return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(last_layer_options[0]))
    model.add(Activation(last_layer_options[1]))
    return model


def main():
    print('This module is used for defining models.')


if __name__ == '__main__':
    main()
