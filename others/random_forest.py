
"""
@author: Sreenivas Bhattiprolu
First part fo code: Same code as logistic regression. 
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#STEP 1: DATA READING AND UNDERSTANDING
#匯入csv檔
df = pd.read_csv("testall.csv")

#STEP 2: DROP IRRELEVANT DATA
#In our example, Images_Analyzed reflects whether it is good analysis or bad
#so should not include it. ALso, User number is just a number and has no inflence
#on the productivity, so we can drop it.
#去除與結果不相關的欄位
df.drop(['file'], axis=1, inplace=True)
print(df.head())

#STEP 5: PREPARE THE DATA.
#確保檢查結果資料型態Y
#Y is the data with dependent variable, this is the Productivity column
Y = df["people"].values  #At this point Y is an object not of type int
#Convert Y to int
Y=Y.astype('int')

#製作檢查data
#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ["people"], axis=1)  
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
#如果沒設random_state每次隨機取樣結果將不同
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

#STEP 7: Defining the model and training.

# Import the model we are using
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 
#Let us use classifier since this is a classification problem

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
# Train the model on training data
model.fit(X_train, y_train)


#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
#印出被判斷與判斷後的結果
prediction_test = model.predict(X_test)
print(y_test)
print(prediction_test)


from sklearn import metrics
#檢查準確度
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
