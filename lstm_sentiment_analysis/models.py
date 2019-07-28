# -*- coding: utf-8 -*-
'''
This module is used for defining models.

@Time    :   2019
@Author  :   ZHOU, YANG
@Contact :   yzhou0000@gmail.com
'''

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, concatenate
from keras.layers import LSTM, GRU


def get_last_layer_options(num_classes):
    """获取输出层的units数量、激活函数、损失函数。

    Parameters
    ----------
        num_classes: int, 类别数量
    Returns
    -------
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


def text_cnn_model(options_last_layer,
                   num_features=10000,
                   sequence_len=256,
                   embedding_dim=128,
                   filters=2,
                   pool_size=3,
                   dropout_rate=0.2,
                   fc_units=32,
                   use_pretrained_embedding=False,
                   is_embedding_trainable=False,
                   embedding_matrix=None):
    """创建一个Text-CNN模型的实例。

    Parameters
    ----------
        options_last_layer: tuple, 输出层的units数量、激活函数、损失函数
        num_features: int, 词汇表大小，即embedding层的输入维度
        sequence_len: int, 输入序列的长度
        embedding_dim: int, 词向量维度，即embedding层的输出维度
        filters: int, 滤波器数量，即channel数量
        pool_size: int, 池化层核宽。
        dropout_rate: float, Dropout层的drop比例
        fc_units: int, 输出层前的全连接层的units数量
        use_pretrained_embedding: bool, 是否使用预训练embedding
        is_embedding_trainable: bool, true if embedding layer is trainable
        embedding_matrix: dict, dictionary with embedding coefficients
    Returns
    -------
        model: 一个Text-CNN模型实例
    """
    inputs = Input(shape=(sequence_len,))
    if use_pretrained_embedding:
        outputs = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=sequence_len,
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable)(inputs)
    else:
        outputs = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=sequence_len)(inputs)
    cnns = []
    for size in (2, 3, 4):
        cnn = Conv1D(filters=filters,
                     kernel_size=size,
                     padding='same',
                     strides=1)(outputs)
        cnn = Activation('relu')(cnn)
        cnn = MaxPooling1D(pool_size=pool_size,
                           padding='valid')(cnn)
        cnns.append(cnn)
    outputs = concatenate(cnns, axis=1)

    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)

    outputs = Dense(fc_units)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dense(options_last_layer[0])(outputs)
    outputs = Activation(options_last_layer[1])(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def lstm_model(last_layer_options,
               num_features=10000,
               embedding_dim=128,
               sequence_len=256,
               lstm_units=64,
               dropout_rate=0.2,
               fc_units=32,
               use_pretrained_embedding=False,
               embedding_matrix=None,
               is_embedding_trainable=False):
    """创建一个LSTM模型的实例。

    Parameters
    ----------
        last_layer_options: tuple, 输出层的units数量、激活函数、损失函数
        num_features: int, 词汇表大小，词汇表大小，即embedding层的输入维度
        embedding_dim: int, 词向量维度，即embedding层的输出维度
        sequence_len: int, 输入序列的长度
        lstm_units: int, LSTM层的unit数量
        dropout_rate: float, Dropout层的drop比例
        fc_units: int, 输出层前的全连接层的units数量
        use_pretrained_embedding: bool, 是否使用预训练的embedding
        is_embedding_trainable: bool, true if embedding layer is trainable
        embedding_matrix: dict, dictionary with embedding coefficients
    Returns
    -------
        model: 一个LSTM模型实例
    """
    model = Sequential()
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

    model.add(Dense(fc_units))
    model.add(Activation('relu'))
    model.add(Dense(last_layer_options[0]))
    model.add(Activation(last_layer_options[1]))
    return model


def gru_model(options_last_layer,
              num_features=10000,
              sequence_len=256,
              embedding_dim=128,
              gru_units=64,
              dropout_rate=0.2,
              fc_units=32,
              use_pretrained_embedding=False,
              is_embedding_trainable=False,
              embedding_matrix=None):
    """创建一个GRU模型的实例。

    Parameters
    ----------
        options_last_layer: tuple, 输出层的units数量、激活函数、损失函数
        num_features: int, 词汇表大小，即embedding层的输入维度
        sequence_len: int, 输入序列的长度
        embedding_dim: int, 词向量维度，即embedding层的输出维度
        gru_units: int, GRU层的units数量
        dropout_rate: float, Dropout层的drop比例
        fc_units: int, 输出层前的全连接层的units数量。
        use_pretrained_embedding: bool, 是否使用预训练的embedding
        is_embedding_trainable: bool, true if embedding layer is trainable
        embedding_matrix: dict, dictionary with embedding coefficients
    Returns
    -------
        model: 一个GRU模型实例。
    """
    inputs = Input(shape=(sequence_len,))
    if use_pretrained_embedding:
        outputs = Embedding(input_dim=num_features,
                            input_length=sequence_len,
                            output_dim=embedding_dim,
                            trainable=is_embedding_trainable,
                            weights=[embedding_matrix])(inputs)
    else:
        outputs = Embedding(input_dim=num_features,
                            input_length=sequence_len,
                            output_dim=embedding_dim,
                            embeddings_initializer='uniform')(inputs)
    outputs = GRU(units=gru_units,
                  activation='tanh',
                  recurrent_activation='hard_sigmoid',
                  kernel_initializer='glorot_uniform',
                  use_bias=True,
                  bias_initializer='zeros',
                  dropout=0.2,
                  recurrent_dropout=0.2,
                  return_sequences=True)(outputs)

    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)

    outputs = Dense(fc_units)(outputs)
    outputs = Activation('relu')(outputs)
    last_units, last_activation = options_last_layer[:2]
    outputs = Dense(last_units)(outputs)
    outputs = Activation(last_activation)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def main():
    print('This module is used for defining models.')


if __name__ == '__main__':
    main()
