# -*- coding: utf-8 -*-
'''
@Time : 2019
@Author : ZHOU, YANG 
@Contact : yzhou0000@gmail.com
'''

import os
import pickle

from keras.models import load_model


init_path = os.getcwd()
save_dir = os.path.join(init_path, 'saved_models')
model_name = ''
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, 'tokenizer.pickle')
