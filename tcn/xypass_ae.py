# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:50:24 2022

@author: a3311
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Conv1D, Input, Flatten, Dense, concatenate, LSTM, AveragePooling1D, Conv1DTranspose, UpSampling1D, MaxPooling1D
from keras.layers import Conv2D, ReLU, BatchNormalization, Activation, MaxPooling1D
from keras import optimizers
from keras.layers.core import Dropout
from keras.models import Model
from joblib import load
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

import data_processor as data_processor
import time



def dense_factor(inputs):
    h_1 = BatchNormalization()(inputs)
    h_1 = Activation('relu')(h_1)
    output = Conv1D(32, 3, padding='same')(h_1)
    return output

def dense_block(inputs):
    concatenated_inputs = inputs

    for i in range(3):
        x = dense_factor(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, x])

    return concatenated_inputs



def transition_block(inputs):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv1D(128, 1, padding='same')(x)
    x = MaxPooling1D(2)(x)

    return x



def build_train_model():
    # load data
    
    tcndata=np.loadtxt('./tcndata/tcn_xyall.txt', dtype=float)
    print(tcndata.shape)
    xtrain = [[[0 for k in range(2)] for j in range(28)] for i in range(len(tcndata))]
    for i in range (len(tcndata)):
        for j in range (28):
            odd=j*2
            even=j*2+1
            xtrain[i][j][0]=tcndata[i][odd]
            xtrain[i][j][1]=tcndata[i][even]
    x_train=np.array(xtrain)
    print(x_train.shape)
    
    timesteps = x_train.shape[1]
    input_dim = x_train.shape[2]

    inputs = Input(shape=(timesteps, input_dim))
    conv_1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    avgpooling_1 = MaxPooling1D(pool_size=2)(conv_1)
    denseblock_1 = dense_block(avgpooling_1)
    transition_1 = transition_block(denseblock_1)
    flat_layer = Flatten()(transition_1)
    denseblock_2 = dense_block(transition_1)
    conv_transpose_1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(denseblock_2)
    conv_transpose_1 = BatchNormalization()(conv_transpose_1)
    avgpooling_2 = UpSampling1D(2)(conv_transpose_1)
    conv_transpose_2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(avgpooling_2)
    conv_transpose_2 = BatchNormalization()(conv_transpose_2)
    avgpooling_3 = UpSampling1D(2)(conv_transpose_2)
    conv_transpose_3 = Conv1D(filters=2, kernel_size=3,activation='sigmoid', padding='same')(avgpooling_3)
    conv_transpose_3 = BatchNormalization()(conv_transpose_3)
    autoencoder = Model(inputs, conv_transpose_3)

    encoder = Model(inputs, flat_layer)

    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    print(autoencoder.summary())

    result = autoencoder.fit(x_train, x_train, epochs=100, batch_size=128, verbose=1, validation_data=(x_train, x_train))
    
    loss, mse = autoencoder.evaluate(x_train, x_train, verbose=2)

    print('Loss: %f, mse: %f' % (loss, mse))

    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('autoencoder')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    p = autoencoder.predict(x_train)
    print(p.shape)
    
    return p

tcndata=build_train_model()

outfile = open("ae_xyall.txt", 'w')
ae_txt=[[0]*56 for i in range(3600)]
for i in range (len(tcndata)):
    for j in range (28):
        odd=j*2
        even=j*2+1
        ae_txt[i][odd]=tcndata[i][j][0]
        ae_txt[i][even]=tcndata[i][j][1]
        outfile.write(str(ae_txt[i][odd])+' '+str(ae_txt[i][even]) +' ')
    outfile.write('\n')
ae_txt=np.array(ae_txt)
outfile.close()
print(ae_txt.shape)