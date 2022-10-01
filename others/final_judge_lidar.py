#!/usr/bin/env python3

import sys
import math
import numpy as np
from rplidar import RPLidar
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

#for st-gcn
import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from model import *
from metric import accuracy
from config import get_args
import warnings
warnings.filterwarnings("ignore")
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#for gui
from tkinter import *
from tkinter.constants import CENTER #加到第一行
from PIL import Image,ImageTk
import os
import numpy as np
import cv2


PORT_NAME = 'com3'
i=0
coordinate=[[0]*2 for i in range(720)]
onecircle=0

def graph(y):
    global onecircle
    
    plt_order = []
    
    np.shape(y) #功能是讀取矩陣的形狀，輸出如: (6791, 2) 
    #coordinate=[[0]*2 for i in range(720)] #二維矩陣，每一row是元素為兩個0的一維矩陣, 有720個row
    
    count=0 #整個txt的pointer
    onecircle=0 #720個資料算一個動作
    graphnum=0 #計算有多少個動作
    List_size_in_graph=[] #紀錄被判定成人的群有幾個點
    total_node_size=0 #紀錄所有人的點數的總和，用來設定array大小
    check_for_peop=0 #用來判斷此動是否有被判定為人的物件
    stgcn_data=[[[0]*2]*28]*15
    
    #判斷前一動人物軌跡變數
    adjacency_pointx = []
    adjacency_pointy = []
    adj_index = 0
    last_center_x = 0
    last_center_y = 0
    closest = 0
    human_index= [] #紀錄沒有被判斷到人的照片數，第一章照片為0
    tcndata_count=0
    
    #極座標轉換成座標
    if (onecircle < 720):
        if(float(y[2])>0 and float(y[2])<90):
            coordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))
            coordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))

        if(float(y[2])>90 and float(y[2])<180):
            coordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))
            coordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))

        if(float(y[2])>180 and float(y[2])<270):
            coordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))
            coordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))

        if(float(y[2])>270 and float(y[2])<360):
            coordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))
            coordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))

        onecircle+=1
    else:
        
        graphnum=graphnum+1
        x=np.array(coordinate)
        clustering=DBSCAN(algorithm='brute',eps=180,leaf_size=180,metric='euclidean'
                          ,metric_params=None,min_samples=2,n_jobs=1,p=None).fit(coordinate)
        onecircle=0 #清空數據指針
        #整體房間空間
        plt.figure(figsize=(10,10),dpi=250)
        plt.scatter(x[:,0],x[:,1],c=clustering.labels_)
        plt.show()
        clustering.fit(x)
        Y_pred = clustering.labels_ #分群狀況
       
        
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
        
        '''
        #印出每一個物件
        for m in range(0,maxcluster):
            plt.scatter(arrx[m],arry[m])
            plt.show()
        '''
        
        #得到rf tree的資料
        length=np.zeros(maxcluster) #y的距離差長度
        width=np.zeros(maxcluster) #x的距離差寬度
        group_number = len(Listx) #總共幾群
        center_distance=np.zeros(maxcluster)
        center_x=np.zeros(maxcluster) #y的距離差長度
        center_y=np.zeros(maxcluster) #x的距離差寬度

        #計算長寬
    
        for m in range(maxcluster):
            length[m] = max(Listx[m]) - min(Listx[m])
            width[m] = max(Listy[m]) - min(Listy[m])

        #計算到中心點的平均距離
       
        for m in range(maxcluster):
            center_x = sum(Listx[m])/len(Listx[m])
            center_y = sum(Listy[m])/len(Listy[m])
      
        for n in range(len(Listx[m])):           
            center_distance[m]=center_distance[m]+ ((Listx[m][n]-center_x)**2+ (Listy[m][n]-center_y)**2)**0.5  
      
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
        

        #轉換rf tree 訓練樣本
        sheet=[[0 for _ in range(8)] for _ in range(group_number)] #group_number個裡面有8個0的一維陣列
        for i in range(group_number):
            sheet[i][0]=num_of_each_group[i]
            sheet[i][1]=x_std[i]
            sheet[i][2]=y_std[i]
            sheet[i][3]=width[i]
            sheet[i][4]=length[i]
            sheet[i][5]=center_distance[i]
            sheet[i][6]=x_var[i]
            sheet[i][7]=y_var[i]
        sheet = pd.DataFrame(sheet)
          
        #匯入rf tree 檔案
        df = pd.read_csv("data0701.csv")
        #去除與決策不相關之數據
        df.drop(['file'], axis=1, inplace=True)
        #確認結果資料型態
        Y = df["people"].values 
        Y=Y.astype('int')
        #創造訓練條件
        X = df.drop(labels = ["people"], axis=1)  
        
        from sklearn.model_selection import train_test_split
        #如果沒設random_state每次隨機取樣結果將不同
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
        from sklearn.ensemble import RandomForestClassifier
        #model製作
        model = RandomForestClassifier(n_estimators = 10, random_state = 30)
        model.fit(X_train, y_train)
        prediction_test = model.predict(sheet)
        #print(prediction_test)
        
        #sheet為此張圖片判斷出的物件特徵 prediction為 rf tree 判斷的結果
        sheet=sheet.values
        check_for_peop=0
        fifty=0
        
        #前一動判別
        prediction_test_people = np.where(prediction_test == 1)
        prediction_test_people = prediction_test_people[0]
        #print("被預測為人的群的index:  ",prediction_test_people)
        #print("被預測為人的群的數量", len(prediction_test_people))
        
        if(len(prediction_test_people)>1): #若有多個被標記為人
            #print("  多個為人")
            if(last_center_x!=0 and last_center_y!=0): #有上一個人
                distance_with_previous = []
                for temp in (prediction_test_people):
                    
                    recent_center_x = sum(Listx[temp])/len(Listx[temp])
                    recent_center_y = sum(Listy[temp])/len(Listy[temp])
                    distance_with_previous.append(((recent_center_x-last_center_x)**2 + (recent_center_y-last_center_y)**2)**0.5) 
                    

                index = distance_with_previous.index(min(distance_with_previous))
                closest =  prediction_test_people[index]

                check_for_people = 1
            else:
                check_for_people = 1
        elif(len(prediction_test_people)==1):
            
            closest =  prediction_test_people[0]
            check_for_people = 1
        elif(len(prediction_test_people)==0):
            
            check_for_people = 0

        if(check_for_people == 1):    
            List_size_in_graph.append(sheet[closest][0])  #把人的size放進array中
            
            total_node_size=total_node_size+sheet[closest][0] #計算所有人物件的總數
            check_for_peop=1
            adjacency_pointx.append([])
            adjacency_pointy.append([])
            for i, j in zip(Listx[closest],Listy[closest]):
                adjacency_pointx[adj_index].append(i) 
                adjacency_pointy[adj_index].append(j) 
            
            human_index.append(adj_index)
            adj_index =  adj_index+1
            last_center_x = sum(Listx[closest])/len(Listx[closest])
            last_center_y = sum(Listy[closest])/len(Listy[closest])
        else:
            human_index.append(-1)
            
        
        if(tcndata_count<15):
            if(human_index[tcndata_count]==-1):
                while (fifty<28):
                    stgcn_data[tcndata_count][fifty][0]=0
                    stgcn_data[tcndata_count][fifty][1]=0
                    
                    fifty=fifty+1
                
            else:
                while (fifty<28):
                    if(fifty<len(adjacency_pointx[human_index[tcndata_count]])):
                        stgcn_data[tcndata_count][fifty][0]=adjacency_pointx[human_index[tcndata_count]][fifty]
                        stgcn_data[tcndata_count][fifty][1]=adjacency_pointy[human_index[tcndata_count]][fifty]
                        #tcndata.write(str(adjacency_pointx[human_index[tcndata_count]][fifty])+' '+str(adjacency_pointy[human_index[tcndata_count]][fifty])+' ')
                    else:
                        stgcn_data[tcndata_count][fifty][0]=0
                        stgcn_data[tcndata_count][fifty][1]=0
                    fifty=fifty+1
                #種類標記
            tcndata_count=tcndata_count+1
        else:
            stgcn_data = torch.from_numpy(np.asarray(stgcn_data)).to(device)
            test_loader  = data.DataLoader(data.TensorDataset(stgcn_data.to(device)),
                                           batch_size = args.batch_size, shuffle=False)
            model.load_state_dict(torch.load(os.path.join(args.model_path, 
        												  'model-%d.pkl'%(50))))
            for i, x in enumerate(test_loader):
                logit = model(x[0].float())
                all_predict = torch.cat((all_predict, logit), dim=0)
    			#print(F.softmax(logit, 1).cpu().numpy(), torch.max(logit, 1)[1].float().cpu().numpy())
                target = test_label[i] #答案
                np_target=target.cpu().numpy()
                print(np_target)
                
            tcndata_count=0



        
        
def run(path):  
    
    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    outfile = open("route_five40.txt", 'w')
    try:
        print('Recording measurments... Press Crl+C to stop.')
        for measurment in lidar.iter_measures():
            line = '\t'.join(str(v) for v in measurment)
            y=line.split('\t')
            outfile.write(y[2]+' '+y[3] + '\n')
            #print(y)
            graph(y)
    except KeyboardInterrupt:
        print('Stoping.')
    lidar.stop()
    lidar.disconnect()
    outfile.close()

if __name__ == '__main__':
    
    run(sys.argv[0])