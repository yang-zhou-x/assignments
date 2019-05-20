# -*- coding: utf-8 -*-
'''
@Time    :   2019
@Author  :   ZHOU, YANG
@Contact :   yzhou0000@gmail.com
'''

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

import preprocess as pp
import models as m


# Parameters
seed = 2019
num_classes = 2  # 类别数量
max_features = 10000  # 词汇表的单词总数
embedding_dim = 128  # 词向量的维度
sequence_len = 30  # 序列的长度
lstm_units = 64  # LSTM的输出维度
dropout_rate = 0.2
batch_size = 64
optimizer = Adam(lr=.002, epsilon=.001)
epochs = 50  # 实际epoch<=50
metric = ['accuracy']
model_name = 'lstm_shpping.hdf5'
tokenizer_name = 'tokenizer.pickle'

# Paths
init_path = os.getcwd()
data_dir = os.path.join(init_path, 'datasets')
train_test_path = os.path.join(data_dir, 'online_shopping_10_cats.csv')
stopwords_path = os.path.join(data_dir, 'cn_stopwords_punctuations.csv')
save_dir = os.path.join(init_path, 'saved_models')
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, tokenizer_name)

# Model
last_layer_options = m.get_last_layer_options(num_classes)
model = m.lstm_model(num_features=max_features,
                     embedding_dim=embedding_dim,
                     sequence_len=sequence_len,
                     lstm_units=lstm_units,
                     dropout_rate=dropout_rate,
                     last_layer_options=last_layer_options)
model.summary()
model.compile(optimizer=optimizer,
              loss=last_layer_options[2],
              metrics=metric)

# Data
df_data = pd.read_csv(train_test_path, header=0)
print(f'原始数据集的形状为: {df_data.shape}')
df_data['review'] = df_data['review'].astype('str')
print('计算每个类别评论长度的中位数、平均值、标准差：')
df_data['len'] = df_data['review'].map(lambda x: len(x))
print(df_data[['cat', 'len']]
      .groupby(by='cat')
      .agg(['median', 'mean', 'max', 'min', 'std']))

df_data = pp.reduce_memory_usage(df_data)
stopwords = pp.get_stopwords(stopwords_path)
x = pp.tokenize_texts(df_data['review'].values, stopwords)
y, _ = pp.encode_y(df_data['label'].values, num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=seed)

x_train, x_test, tokenizer = pp.vectorize_texts_to_sequences(
    x_train, sequence_len, dict_size=max_features, x_test=x_test)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Train
callback = [EarlyStopping('val_acc', patience=5),
            ModelCheckpoint(model_path, 'val_acc', verbose=1,
                            save_best_only=True)]
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callback,
                    validation_data=(x_test, y_test))


# Plot
def plot_acc_loss(history):
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
plot_acc_loss(history)
