import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time

from model import *
from metric import accuracy
from config import get_args


from sklearn.metrics import confusion_matrix
#from resources.plotcm import plot_confusion_matrix
from plotcm import plot_confusion_matrix

args = get_args() #參數

#device是作為Tensor或Model被分配的位置(torch.Tensor分配到device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load 訓練、驗證、測試的data和label
train_tensor, train_label = torch.load(args.train_path)
valid_tensor, valid_label = torch.load(args.valid_path)
test_tensor , test_label  = torch.load(args.test_path)

'''
要對大量數據進行load和處理時因為可能會出現記憶體不夠用的情況，
這時候就需要用到數據集類Dataset或TensorDataset和數據集加載類DataLoader了。
使用這些類後可以將原本的數據分成小塊，在需要使用的時候再一部分一本分讀進記憶體中，
而不是一開始就將所有數據讀進記憶體中。

TensorDataset本質上與python zip方法類似，對數據進行打包整合。

a = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
b = torch.tensor([1,2,3,4])
train_data = TensorDataset(a,b)
print(train_data[0:4])

output:
(tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]]), tensor([1, 2, 3, 4]))

DataLoader本質上就是一個iterable（跟python的內置類型list等一樣），
並利用多進程來加速batch data的處理，使用yield來使用有限的內存。

a = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
a = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
b = torch.tensor([1,2,3,4])
train_data = TensorDataset(a,b)
data = DataLoader(train_data, batch_size=2, shuffle=False)
for i, j in enumerate(data):
    x, y = j
    print(' batch:{0} x:{1}  y: {2}'.format(i, x, y))

output:
 batch:0 x:tensor([[1, 1, 1],
        [2, 2, 2]])  y: tensor([1, 2])
 batch:1 x:tensor([[3, 3, 3],
        [4, 4, 4]])  y: tensor([3, 4])
'''
#data.TensorDataset規定傳入的數據必須是torch.Tensor類型的
print(train_tensor.shape) #torch.Size([172, 32, 15, 3]) #影片數, 一個影片幾禎, 15個關節, x、y、置信度
print(valid_tensor.shape) #torch.Size([20, 32, 15, 3])
print(test_tensor.shape) #torch.Size([23, 32, 15, 3])
train_loader = data.DataLoader(data.TensorDataset(train_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False) #rgs.batch_size: 1
valid_loader = data.DataLoader(data.TensorDataset(valid_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
test_loader  = data.DataLoader(data.TensorDataset(test_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)

train_label = train_label.type(torch.LongTensor) #加這一行，不然會有RuntimeError: expected scalar type Long but found Int
train_label = train_label.to(device) #將tensor變數copy一份到device所指定的cpu上，之後的運算都在cpu上進行
#print(train_label.shape) #torch.Size([172])
valid_label = valid_label.type(torch.LongTensor) #加這一行
valid_label = valid_label.to(device)
test_label = test_label.type(torch.LongTensor) #加這一行
test_label  = test_label.to(device)




A = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
A = torch.from_numpy(np.asarray(A)).to(device) #將tensor變數copy一份到device所指定的cpu上，之後的運算都在cpu上進行




#output : 3
model = GGCN(A, train_tensor.size(3), args.num_classes, #args.num_classes: 9
			 [train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64],
			 args.feat_dims, args.dropout_rate)
			#args.feat_dims:13,　args.dropout_rate：0.0
if device == 'cuda':
	model.cuda()

num_params = 0


'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
    def forward(self, x):
        out = self.linear(x)
        return out
net = Net()

output: 
for p in net.parameters():
    print(p)
Parameter containing:是三乘五！！
tensor([[ 0.3853, -0.1898, -0.1759,  0.3923,  0.3055],
        [-0.1601,  0.1493, -0.1094, -0.2387, -0.0621],
        [ 0.0700,  0.1590,  0.2732,  0.4451, -0.2211]], requires_grad=True)
Parameter containing:
tensor([ 0.0837,  0.0941, -0.0212], requires_grad=True)
'''
#model.parameters()會返回一個生成器（迭代器），生成器每次生成的是Tensor類型的數據．
#算參數數量
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params)) #基本語法是通過 {} 和 : 來代替以前的 % 。format 函數可以接受不限個參數，位置可以不按順序。這邊是不設置指定位置，按默認順序

criterion = nn.CrossEntropyLoss()
#model.parameters()與model.state_dict()是Pytorch中用於查看網絡參數的方法。一般來說，前者多見於優化器的初始化，例如：
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, #learning_rate = 0.005
					   betas=[args.beta1, args.beta2], weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.1) #pytorch中六種學習率調整方法的其中一種，詳見如下
'''
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
功能： 等間隔調整學習率，調整倍數為gamma倍，調整間隔為step_size。間隔單位是step。需要注意的是，step通常是指epoch，不要弄成iteration了。

參數：
step_size(int)- 學習率下降間隔數，若為30，則會在30、60、90......個step時，將學習率調整為lr*gamma。
gamma(float)- 學習率調整倍數，默認為0.1倍，即下降10倍。
last_epoch(int)- 上一個epoch數，這個變量用來指示學習率是否需要調整。當last_epoch符合設定的間隔時，就會對學習率進行調整。當為-1時，學習率設置為初始值。
'''

best_epoch = 0
best_acc = 0
def train():
	global best_epoch, best_acc

	#用來斷點繼續訓練用的
	#如果對模型進行了訓練，例如直到Epoch 42並終止過程，如果給出了“ -start_epoch = 42”選項，則可以將其設置為從Epoch 42開始
	if args.start_epoch:#通常不會跑進這個if
		model.load_state_dict(torch.load(os.path.join(args.model_path, 
													  'model-%d.pkl'%(args.start_epoch))))

	# Training
	for epoch in range(args.start_epoch, args.num_epochs): #range: 0, 30 (所以實值為0~29)
		train_loss = 0
		train_acc  = 0
		scheduler.step() #對優化器的學習率進行調整
		model.train() #要將模型從評估模式轉為訓練模式，可以使用
		for i, x in enumerate(train_loader): #i是0, 1, 2....171這樣依序增加的，x是tensor陣列, 每個epoch有172筆訓練資料要跑
			'''
			print((np.array(x[0])).shape)
			output:
			(1, 32, 15, 3) # i = 0
			(1, 32, 15, 3) # i = 1
			.
			.
			(1, 32, 15, 3) # i = 171
			共172行
			只有x[0]，沒有x[1]
			'''

			#Logit的一個很重要的特性就是沒有上下限——這就給建模帶來極大方便
			logit = model(x[0].float()) #概率P的變化範圍是[0, 1]，而Odds的變化範圍是[0. 正無限]。再進一步，如果對Odds取自然對數，就可以將概率P從範圍[0, 無限]映射到[正無限, 負無限]。 Odds的對數稱之為Logit。
			target = train_label[i] #train_label是tensor([5,..., 1])，共172個label, 取第i個其實也只有1個

			loss = criterion(logit, target.view(1)) #view 相當於numpy中resize（）的功能，但是用法可能不太一樣。

			model.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			train_acc  += accuracy(logit, target.view(1))

		print('[epoch',epoch+1,'] Train loss:',train_loss/(i+1), 'Train Acc:',train_acc/(i+1)) #注意

		if (epoch+1) % args.val_step == 0:
			model.eval() #要將模型從訓練模式轉為評估模式，則可以使用
			val_loss = 0
			val_acc  = 0
			with torch.no_grad():
				for i, x in enumerate(valid_loader):
					logit = model(x[0].float())
					target = valid_label[i]

					val_loss += criterion(logit, target.view(1)).item()
					val_acc += accuracy(logit, target.view(1))

				if best_acc <= (val_acc/i):
					best_epoch = epoch+1
					best_acc = (val_acc/i)
					torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

			print('Val loss:',val_loss/(i+1), 'Val Acc:',val_acc/(i+1))


@torch.no_grad()
def get_all_preds(model, loader):
	all_preds = torch.tensor([]) #all_preds來保存輸出預測
	for batch in loader: #迭代來自數據加載器的批處理
		images, labels = batch

		preds = model(images)
		all_preds = torch.cat( #並將輸出預測與all_preds張量連接在一起
			(all_preds, preds)
			, dim=0
		)
	return all_preds #最後，所有預測all_preds將返回給調用方

def test():
	global best_epoch

	#model.load_state_dict(torch.load(os.path.join(args.model_path, 
												  #'model-%d.pkl'%(50))))
	model.load_state_dict(torch.load('./models\model-50.pkl'))
	print("load model from 'model-%d.pkl'"%(best_epoch))

	model.eval() #要將模型從訓練模式轉為評估模式，則可以使用
	test_loss = 0
	test_acc  = 0
	print("start")
	with torch.no_grad():
		start = time.time()
		global all_predict
		all_predict = torch.tensor([])
		for i, x in enumerate(test_loader):
			logit = model(x[0].float())
			all_predict = torch.cat((all_predict, logit), dim=0)
			#print(F.softmax(logit, 1).cpu().numpy(), torch.max(logit, 1)[1].float().cpu().numpy())
			target = test_label[i] #答案
			np_target=target.cpu().numpy()
			#print(np_target)
			test_loss += criterion(logit, target.view(1)).item()
			test_acc  += accuracy(logit, target.view(1))
			#print(target)
			#print(logit.argmax())
		print(all_predict)
		print("The time used to execute this is given below")
		end = time.time()
		print(end - start)
    
		stacked = torch.stack(
			(
				test_label
				, all_predict.argmax(dim=1)
			)
			, dim=1
		)
		#print(stacked)
		#print(stacked.shape)
		cmt = torch.zeros(6, 6, dtype=torch.int64)
		for p in stacked:
			tl, pl = p.tolist()
			cmt[tl, pl] = cmt[tl, pl] + 1
		#print(cmt)
		cm = confusion_matrix(test_label, all_predict.argmax(dim=1))
		names = ('abnormal', 'route1', 'route2', 'route3', 'route4', 'route5')
		plt.figure(figsize=(12, 12))
		plot_confusion_matrix(cm, names)
		plt.xticks(rotation=0)
		plt.show()
	print('Test loss:',test_loss/(i+1), 'Test Acc:',test_acc/(i+1))

if __name__ == '__main__':
	'''if args.mode == 'train':
		train()
	elif args.mode == 'test':
		best_epoch = args.test_epoch'''
	best_epoch = 50
	test()

