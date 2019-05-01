# -*- coding: utf-8 -*-
'''
@Time : 2019/04/30 15:52
@Author : ZHOU, YANG 
@Contact : zhouyang0995@ruc.edu.cn
'''

import os
import pickle

from keras.models import load_model


init_path = os.getcwd()
save_dir = os.path.join(init_path, 'saved_models')
model_name = ''
model_path = os.path.join(save_dir, model_name)
tokenizer_path = os.path.join(save_dir, 'tokenizer.pickle')
