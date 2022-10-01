
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
    
    tcndata=np.loadtxt('tcnxyall.txt', dtype=float)
    print(tcndata.shape)
    xtrain = [[[0 for k in range(2)] for j in range(28)] for i in range(len(tcndata))]
    for i in range (len(tcndata)):
        for j in range (28):
            odd=j*2
            even=j*2+1
            xtrain[i][j][0]=tcndata[i][odd]/1000
            xtrain[i][j][1]=tcndata[i][even]/1000
    x_train=np.array(xtrain)
    
    
    timesteps = x_train.shape[1]
    input_dim = x_train.shape[2]
    
    
    inputs = Input(shape=(timesteps, input_dim))
    print(inputs.shape)
    conv_1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    print(conv_1.shape)
    avgpooling_1 = MaxPooling1D(pool_size=2)(conv_1)
    print(avgpooling_1.shape)
    denseblock_1 = dense_block(avgpooling_1)
    print(denseblock_1.shape)
    transition_1 = transition_block(denseblock_1)
    print(transition_1.shape)
    flat_layer = Flatten()(transition_1)
    print(flat_layer.shape)
    
    print("ok")
    denseblock_2 = dense_block(transition_1)
    print(denseblock_2.shape)
    conv_transpose_1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(denseblock_2)
    print(conv_transpose_1.shape)
    conv_transpose_1 = BatchNormalization()(conv_transpose_1)
    print(conv_transpose_1.shape)
    avgpooling_2 = UpSampling1D(2)(conv_transpose_1)
    print(avgpooling_2.shape)
    conv_transpose_2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(avgpooling_2)
    print(conv_transpose_2.shape)
    conv_transpose_2 = BatchNormalization()(conv_transpose_2)
    print(conv_transpose_2.shape)
    avgpooling_3 = UpSampling1D(2)(conv_transpose_2)
    print(avgpooling_3.shape)
    conv_transpose_3 = Conv1D(filters=2, kernel_size=3,activation='sigmoid', padding='same')(avgpooling_3)
    print(conv_transpose_3.shape)
    conv_transpose_3 = BatchNormalization()(conv_transpose_3)
    print(conv_transpose_3.shape)
    autoencoder = Model(inputs, conv_transpose_3)
    

    encoder = Model(inputs, flat_layer)

    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    #autoencoder.compile(optimizer='adam',loss='binary_crossentropy')  
    #print(autoencoder.summary())

    result = autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, verbose=1, validation_data=(x_train, x_train))
    
    loss, mse = autoencoder.evaluate(x_train, x_train, verbose=2)

    #print('Loss: %f, mse: %f' % (loss, mse))

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

predict=build_train_model()



      

            
