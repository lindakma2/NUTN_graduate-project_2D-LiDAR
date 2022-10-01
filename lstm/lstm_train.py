# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:53:11 2022

@author: a3311
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from sklearn.utils import shuffle
#from keras.layers import Flatten ,ConvLSTM2D
from keras.layers import Dropout
import time

def create_model():
    '''
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    #隐藏层设置为256, input_shape元组第二个参数1意指features为1
    #下面还有个lstm，故return_sequences设置为True
    model.add(LSTM(units=256,input_shape=(None,56),return_sequences=True))
    model.add(LSTM(units=256))
    #后接全连接层，直接输出单个值，故units为1
    model.add(Dense(units=6))
    model.add(Activation('linear'))
    #model.compile(loss='mse',optimizer='adam')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,input_shape=(5,56)))
    model.add(Dropout(0.5))
    model.add(LSTM(256,return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    return model


tcndata=np.loadtxt('./tcndata/tcn_xyall.txt', dtype=float)
tcnlabel=np.loadtxt('./tcndata/tcn_labelall.txt', dtype=int)



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
            segmentdata[m][even]=tcndata[sliding_position][even]
            segmentdata[m][odd]=tcndata[sliding_position][odd]
        
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

slidind_x, y = shuffle(slidind_x, y, random_state=25)

x_train=slidind_x[144:,:,:]
x_test=y[144:]
y_train=slidind_x[:144,:,:]
y_test=y[:144]

df = pd.read_csv('international-airline-passengers.csv',usecols=['passengers'])



model =create_model()

model.fit(x_train, x_test, batch_size=20,epochs=30,validation_split=0.1)

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