# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:34:48 2016

@author: rflamary
"""

import numpy as np
import scipy as sp

np.random.seed(seed=42)

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, normalization
from keras.layers import Dropout,Flatten, Reshape, concatenate, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D,UpSampling2D, Merge, merge
from keras.utils import np_utils
from keras.layers import Input, Lambda
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint,EarlyStopping, LearningRateScheduler
from keras.models import model_from_json
from keras.engine.topology import Layer
#from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras import objectives

import time
__time_tic_toc=time.time()

def tic():
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    t=time.time()
    return t-__time_tic_toc


def save_model(model,fname='mymodel'):
    model.save_weights(fname+'.h5',overwrite=True)
    open(fname+'.json', 'w').write(model.to_json())

def load_model(fname):
    model = model_from_json(open(fname+'.json').read())
    model.load_weights(fname+'.h5')
    return model


class GlobalAveragePooling0D(Layer):
    """Abstract class for different pooling 1D layers.
    """

    def __init__(self,  
                 border_mode='valid', **kwargs):
        super(GlobalAveragePooling0D, self).__init__(**kwargs)
        if border_mode not in {'valid', 'same'}:
            raise ValueError('`border_mode` must be in {valid, same}.')
        self.border_mode = border_mode
        self.input_spec = [InputSpec(ndim=2)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)

    def _pooling_function(self):
        raise NotImplementedError

    def call(self, x, mask=None):
        #x = K.expand_dims(x, 2)   # add dummy last dimension
        output = K.expand_dims(K.sum(x,axis=1),1)
        return output#K.squeeze(output, 2)  # remove dummy last dimension

    def get_config(self):
        config = {}
        base_config = super(GlobalAveragePooling0D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Clip(keras.constraints.Constraint):


    def __init__(self, m=2):
        self.m = m

    def __call__(self, p):
        desired = K.clip(p, -self.m, self.m)
        return desired

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}


class Select(Layer):
    def __init__(self, sel, **kwargs):
        self.sel = sel
        self.output_dim=sel[1]-sel[0]
        super(Select, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_dim = input_shape[1]
        #self.trainable_weights = []
        pass

    def call(self, x, mask=None):
        return x[self.sel[0]:self.sel[1]]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

        
