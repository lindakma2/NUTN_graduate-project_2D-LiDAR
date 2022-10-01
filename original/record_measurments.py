#!/usr/bin/env python3
'''Records measurments to a given file. Usage example:

$ ./record_measurments.py out.txt'''
import sys
import math
import numpy as np
from rplidar import RPLidar
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


PORT_NAME = 'com3'
i=0
cordinate=[[0]*4 for i in range(720)]
onecircle=0

def graph(y):
    global onecircle
    
    #極座標轉換成座標
    if (onecircle < 719):
        if(float(y[2])>0 and float(y[2])<90):
            cordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))
            cordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))

        if(float(y[2])>90 and float(y[2])<180):
            cordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))
            cordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))

        if(float(y[2])>180 and float(y[2])<270):
            cordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))
            cordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))

        if(float(y[2])>270 and float(y[2])<360):
            cordinate[onecircle][1]=float(y[3])*math.sin(math.radians(float(y[2])))
            cordinate[onecircle][0]=float(y[3])*math.cos(math.radians(float(y[2])))

        onecircle+=1
    else:
        
        #到達3000的時候轉換成圖片然後把onecircle歸0
        x=np.array(cordinate)
        clustering=DBSCAN(algorithm='brute',eps=150,leaf_size=180,metric='euclidean'
                          ,metric_params=None,min_samples=10,n_jobs=1,p=None).fit(cordinate)
        plt.figure(figsize=(10,10),dpi=250)
        plt.scatter(x[:,0],x[:,1],c=clustering.labels_)
        cluster=clustering.labels_
        len(set(cluster))
        print(x ,end='   ')
        onecircle=0
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