from torch.utils import data
import torch
import os
import random
import numpy as np


file_name = './dataset/st-gcn_xyall.txt'
f = open(file_name)
lines = f.readlines()
prev_video = int(lines[0][0]) #第幾部影片
prev_categ = int(lines[0][2])
frames = []
train = []
valid = []
test  = []
train_label = []
valid_label = []
test_label  = []
count=0
for line in lines:
    #print("ok")
    line = line.split( )
    
    
    vid = int(line[0]) #影片數
    aid = int(line[1]) #演員編號
    cid = int(line[2]) #動作編號
    
    
    features = list(map(float, line[3:])) 
    now=np.array(features)
    #print(len(now))
    #norm_val = float(line[-1])
    

    
    if prev_video == vid:
        frames.append(np.reshape(np.asarray(features), (-1,2)))
        
 
    else:
        
        if len(frames) >= 15:
            
            frames = random.sample(frames, 15) #會照順序的抽32個不重複
            frames = torch.from_numpy(np.stack(frames, 0))
    
        else:
            
            frames = np.stack(frames, 0) #把frame轉成np
            xloc = np.arange(frames.shape[0]) #每個影片的大小
            
            new_xloc = np.linspace(0, frames.shape[0], 15) #做0-32之間的等差級數
            #print(new_xloc[1])
            frames = np.reshape(frames, (frames.shape[0], -1)).transpose()
            

            new_datas = []
            #print(data.shape)
            for data in frames:
                new_datas.append(np.interp(new_xloc, xloc, data))  
                count=count+1
            frames = torch.from_numpy(np.stack(new_datas, 0)).t()
            

        frames = frames.view(15, -1, 2)
        if prev_actor < 9:
            train.append(frames)
            train_label.append(prev_categ)
            #train.append(torch.stack([frames[:,:,2],frames[:,:,1],frames[:,:,0]], 2))
            #train_label.append(prev_categ)
            #train.append(torch.stack([frames[:,:,0],frames[:,:,2],frames[:,:,1]], 2))
            #train_label.append(prev_categ)
            #train.append(torch.cat([frames[:,:3,:],frames[:,6:9,:],frames[:,3:6,:],
            #                        frames[:,12:15,:],frames[:,9:12,:]], 1))
            #train_label.append(prev_categ)
        elif prev_actor < 10:
            valid.append(frames)
            valid_label.append(prev_categ)
        else:
            test.append(frames)
            test_label.append(prev_categ)
        frames = [np.reshape(np.asarray(features), (-1,2))]
    prev_actor = aid
    prev_video = vid
    prev_categ = cid
    
if len(frames) >= 15:
    frames = random.sample(frames, 15)
    frames = torch.from_numpy(np.stack(frames, 0))
else:
    frames = np.stack(frames, 0)
    xloc = np.arange(frames.shape[0])
    new_xloc = np.linspace(0, frames.shape[0], 15)
    frames = np.reshape(frames, (frames.shape[0], -1)).transpose()

    new_datas = []
    for data in frames:
        new_datas.append(np.interp(new_xloc, xloc, data))
        
    frames = torch.from_numpy(np.stack(new_datas, 0)).t()
    
    

    
frames = frames.view(15, -1, 2)
if aid < 9:
    train.append(frames)
    train_label.append(prev_categ)
    #train.append(torch.stack([frames[:,:,2],frames[:,:,1],frames[:,:,0]], 2))
    #train_label.append(prev_categ)
    #train.append(torch.stack([frames[:,:,0],frames[:,:,2],frames[:,:,1]], 2))
    #train_label.append(prev_categ)
    #train_label.append(prev_categ)
    #train.append(torch.cat([frames[:,:3,:],frames[:,6:9,:],frames[:,3:6,:],
    #                        frames[:,12:15,:],frames[:,9:12,:]], 1))
    #train_label.append(prev_categ)
elif aid < 10:
    valid.append(frames)
    valid_label.append(prev_categ)
else:
    test.append(frames)
    test_label.append(prev_categ)

print(np.asarray(train_label).shape)
train_label = torch.from_numpy(np.asarray(train_label))
valid_label = torch.from_numpy(np.asarray(valid_label))
test_label  = torch.from_numpy(np.asarray(test_label))

trainlook=np.array(train)

print(train_label.shape)

torch.save((torch.stack(train, 0), train_label), './dataset/train.pkl')
torch.save((torch.stack(valid, 0), valid_label), './dataset/valid.pkl')
torch.save((torch.stack(test, 0),  test_label),  './dataset/test.pkl')
