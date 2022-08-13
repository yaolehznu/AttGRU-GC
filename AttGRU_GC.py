# -*- coding: utf-8 -*-

from __future__ import print_function, division
import keras.backend as K
from keras.layers import Multiply
from keras.layers.core import *
from keras.models import *
import pandas as pd
import numpy as np
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
import os

import matplotlib.pyplot as plt
import copy
import datetime

import scipy.io as sio

def batch_sequence(x, sequence_length):  # 输出对应输入最后时刻+1    数据用作GC分析
    num_points = 10000
    inputs = []
    targets = []
    for p in np.arange(0, num_points, 1):
        # prepare inputs (we're sweeping from left to right in steps sequence_length long)
        start = p
        end = p + sequence_length
        inputs.append(x[start: end, :])
        targets.append(x[end, :])
    inputs = np.array(inputs)
    targets = np.array(targets)
    # idx = np.random.permutation(np.arange(inputs.shape[0]))
    # inputs = inputs[idx]
    # targets = targets[idx]
    return inputs, targets

def batch_sequence2(x, sequence_length):  ## 输出对应输入的最后一个时刻
    num_points = 10000
    inputs = []
    targets = []
    for p in np.arange(0, num_points, 1):
        start = p
        end = p + sequence_length
        inputs.append(x[start: end, :])
        targets.append(x[end-1, :])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def get_activations(model, inputs, layer_name=None):
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations

def attention_3d_block(inputs):  ## 变量级attention，做变量选择
    # inputs.shape = (batch_size, sequence_length, input_dim)
    input_dim = int(inputs.shape[2])
    a_probs = Dense(input_dim, activation='softmax', name='attention_vec1')(inputs)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_3d_block2(inputs, sequence_length):  ## 时序级attention， 做延时估计
    # inputs.shape = (batch_size, sequence_length, input_dim)
    a = Permute((2, 1))(inputs)  # (batch_size, input_dim, sequence_length)
    a = Dense(sequence_length, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec2')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def model_attention_applied_before_gru(seq_len, input_dim):  ## 使用变量级attention
    # K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(seq_len, input_dim,))
    # attention_mul = attention_3d_block(inputs, seq_len)
    attention_mul = attention_3d_block(inputs)
    attention_mul = GRU(num_hidden, return_sequences=False, input_dim=input_dim, W_regularizer=l1(weight_decay), U_regularizer=l1(weight_decay))(attention_mul)
    output = Dense(1)(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_before_gru2(seq_len, input_dim):  ## 使用时序级attention，延时估计 训练预测模型
    # K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(seq_len, input_dim,))
    attention_mul = attention_3d_block2(inputs, seq_len)
    attention_mul = GRU(num_hidden, return_sequences=False, input_dim=input_dim, W_regularizer=l1(weight_decay), U_regularizer=l1(weight_decay))(attention_mul)
    output = Dense(1)(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

sequence_length = 20
batch_size = 2000
num_hidden = 32  ## 调整
num_epoch = 300
data_length = 100000 #4096
weight_decay = 1e-7
mode = 'nonlinear5'

## Prepare data x, y
simulation_name = 'realization_' + mode + '_' + str(data_length) + '.mat'
simulation_data = sio.loadmat(simulation_name)
simulation_data = np.array(simulation_data["data"])
num_channel = simulation_data.shape[1]

data = simulation_data  # 数值都处于[0, 1]之间
x, y = batch_sequence(data, sequence_length=sequence_length)  ## 用作GC分析
xp, yp = batch_sequence2(data, sequence_length=sequence_length)  ## 用作选了预测模型

## GC 分析
granger_matrix = np.zeros([num_channel, num_channel])
var_denominator = np.zeros([1, num_channel])
error_model = []
error_all = []
attention_vector_all = []
hist_result = []
start_time = datetime.datetime.now()

for k in range(num_channel):

    tmp_y = np.reshape(y[:, k], [y.shape[0], 1])
    channel_set = list(range(num_channel))
    input_set = channel_set

    gruatt = model_attention_applied_before_gru(sequence_length, len(input_set))
    rms_prop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-6)
    gruatt.compile(loss='mean_squared_error', optimizer=rms_prop)
    gruatt.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    hist_res = gruatt.fit(x[:, :, input_set], tmp_y, batch_size=batch_size, epochs=num_epoch, verbose=2, validation_split=0.2, callbacks=[early_stopping])
    hist_result.append(hist_res)

    var_denominator[0][k] = np.var(gruatt.predict(x[:, :, input_set]) - tmp_y, axis=0)
    for j in range(num_channel):
        if j not in input_set:
            granger_matrix[j][k] = var_denominator[0][k]
        elif len(input_set) == 1:
            tmp_x = x[:, :, k]
            tmp_x = tmp_x[:, :, np.newaxis]
            granger_matrix[j][k] = np.var(gruatt.predict(tmp_x) - tmp_y, axis=0)
        else:
            tmp_x = x[:, :, input_set]
            channel_del_idx = input_set.index(j)
            tmp_x[:, :, channel_del_idx] = 0
            granger_matrix[j][k] = np.var(gruatt.predict(tmp_x) - tmp_y, axis=0)

granger_matrix = granger_matrix / var_denominator
for i in range(num_channel):
    granger_matrix[i][i] = 1
granger_matrix[granger_matrix < 1] = 1
granger_matrix = np.log(granger_matrix)

M_name = 'GC_Matrix' + '.mat'
gc_dir = 'GC_matrix'
GC_name = os.path.join(gc_dir, M_name)
sio.savemat(GC_name, {'GCMat': granger_matrix})


## 训练预测模型
kk = 6
yo = np.reshape(yp[:, kk], [yp.shape[0], 1])
input_set_p = [0, 1, 2, 5]

gruattT = model_attention_applied_before_gru2(sequence_length, len(input_set_p))
rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
gruattT.compile(loss='mean_squared_error', optimizer=rms_prop)
gruattT.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
hist_res_p = gruattT.fit(xp[:, :, input_set_p], yo, batch_size=batch_size, epochs=num_epoch, verbose=2, validation_split=0.2, callbacks=[early_stopping])

ypp = gruattT.predict(xp[:, :, input_set_p])

### 计算attention矩阵 ###
attention_matrix = np.zeros((sequence_length, len(input_set_p)))
attention_matrixx = []
testing_inputs = xp[:, :, input_set_p]
tst_len = 5000
for i in range(tst_len):
    # print(i)
    testing_inputs_1 = testing_inputs[i, :, :]
    attention_mat = get_activations(gruattT, testing_inputs_1.reshape(1, sequence_length, len(input_set_p)), layer_name='attention_vec2')[0]
    # print(attention_mat)
    attention_matrix = attention_matrix + attention_mat.reshape(sequence_length, len(input_set_p))  # np.mat()
attention_matrixx = (attention_matrix) / tst_len
matrix_name = 'TAttention_Matrix_' + '_' + str(tst_len) + '.mat'
att_dir = 'Tattention_matrix'
attention_name = os.path.join(att_dir, matrix_name)
sio.savemat(attention_name, {'AM': attention_matrixx})


end_time = datetime.datetime.now()
interval = (end_time - start_time).seconds
print('training time: %d seconds' % int(interval))