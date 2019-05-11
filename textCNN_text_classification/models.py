# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG 
@Contact :   yzhou0000@gmail.com
'''

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, concatenate


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


def text_cnn_model(num_features,
                   sequence_len,
                   embedding_dim,
                   filters,
                   pool_size,
                   dropout_rate,
                   fc_units,
                   options_last_layer,
                   use_pretrained_embedding=False,
                   is_embedding_trainable=False,
                   embedding_matrix=None):
    """创建一个Text-CNN模型的实例。
    
    # Parameters
        num_features: int, 词汇表大小，也是embedding层的输入维度
        sequence_len: int, 输入序列的长度
        embedding_dim: int, 词向量维度，即embedding层的输出维度
        filters: int, 滤波器数量，每层的输出维度（channel数量）
        pool_size: int, 池化层核宽。
        dropout_rate: float, Dropout层对于输入的drop比例
        fc_units: int, 输出层前的全连接层的units数量
        options_last_layer: tuple, 输出层的units数量、激活函数、损失函数
        use_pretrained_embedding: bool, 是否使用预训练的embedding
        is_embedding_trainable: bool, true if embedding layer is trainable
        embedding_matrix: dict, dictionary with embedding coefficients
    # Returns
        model: 一个Text-CNN模型实例
    """
    inputs = Input(shape=(sequence_len,))
    # 词嵌入层。可以添加预训练的词向量。
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
    # 卷积层、池化层
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
    # Drop out
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    # 全连接层
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
