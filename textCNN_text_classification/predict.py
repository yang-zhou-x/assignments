# -*- coding: utf-8 -*-
'''
@Time : 2019
@Author : ZHOU, YANG 
@Contact : yzhou0000@gmail.com
'''

import os
import pickle
import numpy as np
from keras.models import load_model

import preprocess as pp


init_path = os.getcwd()
save_dir = os.path.join(init_path, 'saved_models')
model_name = 'THUCNews.hdf5'
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, 'tokenizer.pickle')
stopwords_path = os.path.join(
    init_path, 'datasets/cn_stopwords_punctuations.csv')
data_path = os.path.join(init_path, 'datasets/test.txt')

# load data
texts = []
with open(data_path, encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())

# tokenize
stopwords = pp.get_stopwords(stopwords_path)
texts = pp.tokenize_texts(texts, stopwords)

# to sequences
with open(tokenizer_path) as t:
    tokenizer = pickle.load(t)
x, _ = pp.texts_to_pad_sequences(texts, tokenizer=tokenizer)

# load model
model = load_model(model_path, compile=True)

# predict
y_pred = model.predict(x)
y_pred = np.argmax(y_pred, axis=1)
for p in y_pred:
    print(p)
