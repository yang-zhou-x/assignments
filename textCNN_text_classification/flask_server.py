#! ...
# -*- coding: utf-8 -*-
'''
@Time    : 2019
@Author  : ZHOU, YANG
@Contact : yzhou0000@gmail.com
'''

import os
import pickle
from flask import Flask, request
from keras.models import load_model
from tensorflow import get_default_graph
import preprocess as pp

# Parameters
max_seq_len = 512  # the length of padded sequences
batch_size = 128
model_name = 'xxx.hdf5'
tokenizer_name = 'xxx.pickle'
stopwords_name = 'cn_stopwords_punctuations.csv'
labels_index = 'labels_index.txt'

# Paths
init_path = os.getcwd()

data_dir = os.path.join(init_path, 'datasets')
stopwords_path = os.path.join(data_dir, stopwords_name)

save_dir = os.path.join(init_path, 'saved_models')
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, tokenizer_name)
labels_path = os.path.join(save_dir, labels_index)

graph = get_default_graph()

# stopwords, label names
stopwords = pp.get_stopwords(stopwords_path)
with open(labels_path) as f:
    labels_name = f.readlines()
labels_name = [x.split('\t') for x in labels_name]
labels_name = {int(x[0]): x[1] for x in labels_name}

# model, tokenizer
model = load_model(model_path, compile=True)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)


def get_pred(texts):
    """预测文本类别。

    Parameters
    ----------
        texts: list[str], 原始文本
    Returns
    -------
        y_pred: list[str], 每段文本的预测类别
    """
    texts = pp.tokenize_texts(texts, stopwords)
    texts, _ = pp.texts_to_sequence_vectors(texts, max_seq_len,
                                            tokenizer=tokenizer)
    with graph.as_default():
        y_pred = model.predict(texts,
                               batch_size=batch_size,
                               verbose=0)
    y_pred = np.argmax(y_pred, axis=1).tolist()
    y_pred = [labels_name[x] for x in y_pred]
    return y_pred


# server
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    global graph
    if request.method == 'POST':
        if request.form:  # "form-data"
            try:
                texts = request.form['data']
                texts = eval(texts)
            except KeyError:
                return 'KeyError: Is the key correct?'
            except SyntaxError:
                return 'SyntaxError: Is the format of `value` correct?'
        else:
            data = request.get_data()  # "raw-data"
            try:
                data = eval(data)
                texts = data['data']
            except SyntaxError:
                return 'SyntaxError: Is the format of `value` correct?'
            except KeyError:
                return 'KeyError: Is the key correct?'

        if not texts:
            return 'The `value` is empty.'
        if type(texts[0]) == str:
            y_pred = get_pred(texts)
            res = str(y_pred)
        else:
            return 'Error: Wrong format of `value`.'
        return res  # str
    else:
        return '<h1> Only post requests!</h1>'


@app.route('/user/<name>')
def user(name):
    return '<h1>hello, %s</h1>' % name


if __name__ == '__main__':
    app.run(port=5000,
            host='xxx.xx.x.xx',  # ip address
            debug=False)
