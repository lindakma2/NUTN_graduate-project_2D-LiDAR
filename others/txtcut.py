# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:26:49 2022

@author: a3311
"""

import numpy as np
import numpy
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import csv

from sklearn.datasets import make_blobs
import hdbscan
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

y=np.loadtxt('test (5).txt', dtype=float)

np.shape(y)
coordinate=[[0]*2 for i in range(720)]

count=0
onecircle=0
filenum=0
filename=0
while count < numpy.size(y)/2:  #因為row裡面有兩個col
    if (onecircle < 719):
        if(y[count,0]>0 and y[count,0]<90):
            coordinate[onecircle][0]=y[count][1]*math.cos(math.radians(y[count,0]))
            coordinate[onecircle][1]=y[count][1]*math.sin(math.radians(y[count,0]))
        if(y[count,0]>90 and y[count,0]<180):
            coordinate[onecircle][1]=y[count][1]*math.sin(math.radians(y[count,0]))
            coordinate[onecircle][0]=y[count][1]*math.cos(math.radians(y[count,0]))
        if(y[count,0]>180 and y[count,0]<270):
            coordinate[onecircle][0]=y[count][1]*math.cos(math.radians(y[count,0]))
            coordinate[onecircle][1]=y[count][1]*math.sin(math.radians(y[count,0]))
        if(y[count,0] and y[count,0]<360):
            coordinate[onecircle][1]=y[count][1]*math.sin(math.radians(y[count,0]))
            coordinate[onecircle][0]=y[count][1]*math.cos(math.radians(y[count,0]))
        onecircle+=1
    else:
        x=np.array(coordinate)
        clustering=DBSCAN(algorithm='brute',eps=150,leaf_size=180,metric='euclidean'
                          ,metric_params=None,min_samples=10,n_jobs=1,p=None).fit(coordinate)
        #整體房間空間
        plt.cla()
        plt.figure(figsize=(10,10),dpi=250)
        plt.scatter(x[:,0],x[:,1],c=clustering.labels_)
        plt.show()
        clustering.fit(x)
        Y_pred = clustering.labels_ #分群狀況
        onecircle=0
        #動態陣列
        Listx=[]
        Listy=[]

        #知道有幾個分群
        maxcluster=0
        for k in range(720):
                if(Y_pred[k]>maxcluster):
                    maxcluster=Y_pred[k]
        maxcluster=maxcluster+1
        #print(maxcluster)

        #建立群陣列
        for i in range(maxcluster):
            Listx.append([])
            Listy.append([])

        #把資料分群放進陣列
        for j in range(720):
            if(Y_pred[j]!=-1):
                Listx[Y_pred[j]].append(x[j][0])
                Listy[Y_pred[j]].append(x[j][1])

        arrx=np.array(Listx)
        arry=np.array(Listy)
        
        
        plt.scatter(arrx[1],arry[1])
        
        '''
        for m in range(0,maxcluster):
            plt.scatter(arrx[m],arry[m])
            plt.show()
        '''
        
        #得到rf tree的資料
        maxx=np.zeros(maxcluster)
        maxy=np.zeros(maxcluster)
        minx = np.ones(maxcluster) * 10000000
        miny = np.ones(maxcluster) * 10000000
        length=np.zeros(maxcluster) #y的距離差長度
        width=np.zeros(maxcluster) #x的距離差寬度
        group_number = len(Listx) #總共幾群
        center_distance=np.zeros(maxcluster)

        #計算長寬
        for m in range(maxcluster):
            for n in range(len(Listx[m])):
                if(Listx[m][n]>maxx[m]):
                    maxx[m]=Listx[m][n]
                if(Listy[m][n]>maxy[m]):
                    maxy[m]=Listy[m][n]
                if(Listx[m][n]<minx[m]):
                    minx[m]=Listx[m][n]
                if(Listy[m][n]<miny[m]):
                    miny[m]=Listy[m][n]
            length[m]=maxx[m]-minx[m]
            width[m]=maxy[m]-miny[m]
            #print(length[m],width[m])

        #計算到中心點的平均距離
        for m in range(maxcluster):
            for n in range(len(Listx[m])):
                center_distance[m]=center_distance[m]+ ((Listx[m][n]-(length[m]/2))**2+ (Listy[m][n]-(width[m]/2))**2)**0.5
            center_distance[m]=center_distance[m]/len(Listx[m])

            
        #算每群的點的數量
        num_of_each_group=[] #每群幾個點

        for i in range(group_number):
            num_of_each_group.append(len(Listx[i]))
            
        #算每群的點的標準差
        x_std = [] #每群點的x的標準差
        y_std = [] #每群點的y的標準差
        x_var = [] #每群點的x的變異數
        y_var = [] #每群點的y的變異數

        for i in range(group_number):
            x_std.append(np.std(Listx[i]))
            y_std.append(np.std(Listy[i]))
            x_var.append(np.var(Listx[i]))
            y_var.append(np.var(Listy[i]))
        '''
        #csv輸出
        filename="test1"+"-"+str(filenum)+".csv"
        sheet=[[0 for _ in range(9)] for _ in range(group_number)]
        for i in range(group_number):
            sheet[i][0]=1
            sheet[i][1]=num_of_each_group[i]
            sheet[i][2]=x_std[i]
            sheet[i][3]=y_std[i]
            sheet[i][4]=width[i]
            sheet[i][5]=length[i]
            sheet[i][6]=center_distance[i]
            sheet[i][7]=x_var[i]
            sheet[i][8]=y_var[i]

            #np.savetxt(filename, sheet, delimiter =",",fmt ='% s')
        filenum+=1
        '''
        import pandas as pd
        #轉換rf tree 訓練樣本
        sheet=[[0 for _ in range(8)] for _ in range(group_number)]
        for i in range(group_number):
            sheet[i][0]=num_of_each_group[i]
            sheet[i][1]=x_std[i]
            sheet[i][2]=y_std[i]
            sheet[i][3]=width[i]
            sheet[i][4]=length[i]
            sheet[i][5]=center_distance[i]
            sheet[i][6]=x_var[i]
            sheet[i][7]=y_var[i]
            #np.savetxt("test8.csv", sheet, delimiter =",",fmt ='% s')
        sheet = pd.DataFrame(sheet)
            
        
        import pandas as pd
        from matplotlib import pyplot as plt
        import numpy as np
        
        #匯入rf tree 檔案
        df = pd.read_csv("data0309.csv")
        #去除與決策不相關之數據
        df.drop(['file'], axis=1, inplace=True)
        #確認結果資料型態
        Y = df["people"].values  #At this point Y is an object not of type int
        Y=Y.astype('int')
        #創造訓練條件
        X = df.drop(labels = ["people"], axis=1)  
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
        from sklearn.ensemble import RandomForestClassifier
        #model製作
        model = RandomForestClassifier(n_estimators = 10, random_state = 30)
        model.fit(X_train, y_train)
        prediction_test = model.predict(sheet)
        
        print(prediction_test)
        
        from sklearn import metrics
        #檢查準確度
        #print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))            
    count = count + 1
    






    
    

     
