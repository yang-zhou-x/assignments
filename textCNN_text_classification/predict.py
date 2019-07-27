# -*- coding: utf-8 -*-
'''
This module is used for predicting.

@Time    : 2019
@Author  : ZHOU, YANG 
@Contact : yzhou0000@gmail.com  
'''

import os
import pickle
import numpy as np
from keras.models import load_model
import preprocess as pp

# Parameters
dict_size = 18000
max_sequence_len = 512  # 序列对齐的长度
batch_size = 128

test_name = 'test.txt'
stopwords_name = 'cn_stopwords_punctuations.csv'
model_name = f'TextCNN-{max_sequence_len}-9429.hdf5'
tokenizer_name = f'tokenizer-{dict_size}.pickle'
labels_index = 'labels_index.txt'
outputs_name = 'result.txt'

# Paths
init_path = os.getcwd()

data_dir = os.path.join(init_path, 'datasets')
stopwords_path = os.path.join(data_dir, stopwords_name)
test_path = os.path.join(data_dir, test_name)
outputs_path = os.path.join(data_dir, outputs_name)

save_dir = os.path.join(init_path, 'saved_models')
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, tokenizer_name)
labels_path = os.path.join(save_dir, labels_index)

# load data
texts = []
with open(test_path, encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())

# tokenize
stopwords = pp.get_stopwords(stopwords_path)
texts = pp.tokenize_texts(texts, stopwords)

# to sequences
with open(tokenizer_path, 'rb') as t:
    tokenizer = pickle.load(t)
x, _ = pp.texts_to_sequence_vectors(texts, max_sequence_len,
                                    tokenizer=tokenizer)

# load model
model = load_model(model_path, compile=True)

# predict
y_pred = model.predict(x,
                       batch_size=batch_size,
                       verbose=1)
y_pred = np.argmax(y_pred, axis=1).tolist()

with open(labels_path) as f:
    labels_name = f.readlines()
labels_name = [x.split('\t') for x in labels_name]
labels_name = {int(x[0]): x[1] for x in labels_name}
y_pred = [labels_name[x] for x in y_pred]

# Output
y_pred = [x for x in y_pred]
with open(outputs_path, 'w') as f:
    f.writelines(y_pred)
