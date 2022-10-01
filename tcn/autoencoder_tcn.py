# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:24:54 2022

@author: a3311
"""

#%%
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
    
    xtrain = [[[0 for k in range(2)] for j in range(28)] for i in range(len(tcndata))]
    for i in range (len(tcndata)):
        for j in range (28):
            odd=j*2
            even=j*2+1
            xtrain[i][j][0]=tcndata[i][odd]/1000
            xtrain[i][j][1]=tcndata[i][even]/1000
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


#%%
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN
from sklearn.utils import shuffle 
import random

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, 5 , 56


#tcndata=np.loadtxt('./tcntraindata/tcnxyall.txt', dtype=float)
tcnlabel=np.loadtxt('./tcndata/tcn_labelall.txt', dtype=int)

print(tcndata.shape)

lookback_window = 5
segmentdata=np.zeros((15,56), dtype=int)
slidind_x, y = [], []
sliding_position=0

for j in range (0,len(tcndata),15):
    for m in range (0,15):
        sliding_position=j+m
        for one in range(28):
            odd=one*2+1
            even=one*2
            segmentdata[m][even]=tcndata[sliding_position][one][0]
            segmentdata[m][odd]=tcndata[sliding_position][one][1]
        
    for i in range(lookback_window, 15+1, 2): #range:12~167
        S=segmentdata.copy()
        #slidind_x[j].append(x_train[j][i - lookback_window:i])
        slidind_x.append(S[i - lookback_window:i])
        #print(f'i = {i}')
        #print(f'i - lookback_window = {i - lookback_window}')
        #print(f'x:{slidind_x}\n')
        y.append(tcnlabel[sliding_position])

slidind_x = np.array(slidind_x)
y = np.array(y)
print(slidind_x.shape)
print(y.shape)






tcn_layer = Input(shape=(time_steps, input_dim))
m = TCN()(tcn_layer)
m = Dense(256, activation='relu')(m)
m = Dense(256, activation='relu')(m)
m = Dense(6, activation='softmax')(m)
model = Model(inputs=[tcn_layer], outputs=[m])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
slidind_x, y = shuffle(slidind_x, y, random_state=25)
x_train=slidind_x[144:,:,:]
x_test=y[144:]
y_train=slidind_x[:144,:,:]
y_test=y[:144]
history=model.fit(x_train, x_test, epochs=30, verbose=2)
scores = model.evaluate(slidind_x, y, verbose=0)
#print("Accuracy: %.2f%%" % (scores*100))
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history["loss"])

#%%
start = time.time()
p = model.predict(y_train)

print("The time used to execute this is given below")

end = time.time()

print(end - start)
predict = []

for i in range(len(p)):
    predict_max=max(p[i])
    for j in range(6):
        if(p[i][j]==predict_max):
            predict.append(j)

predict=np.array(predict)
print(np.size(predict))
print(np.size(y_test))

correct=0
for i in range(len(predict)):
    if(y_test[i]==predict[i]):
        correct=correct+1
print(correct)
accuracy=correct/len(y_test)
print("accuracy="+str(accuracy))

#%%
      

            
