# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:19:50 2022

@author: a3311
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN

milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)
print(milk.head())

lookback_window = 12  # 月
milk = milk.values  # 為了簡單起見，這裡保留np陣列

x, y = [], []
for i in range(lookback_window, len(milk)):
    x.append(milk[i - lookback_window:i])
    y.append(milk[i])
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])
model.summary()

model.compile('adam', 'mae')

print('Train...')

model.fit(x, y, epochs=100, verbose=2)

p = model.predict(x)

plt.plot(p)
plt.plot(y)
plt.title('Monthly Milk Production (in pounds)')
plt.legend(['predicted', 'actual'])
plt.show()
