#this script is for testing the model for leaderboard
import numpy as np 
import csv
import torch 
from torch.autograd import Variable

def get_batch2(X,Y,M):
    X,Y = np.asarray(X), np.asarray(Y)
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = torch.FloatTensor(X[batch_indices,:]).type(torch.FloatTensor)
    batch_ys = torch.FloatTensor(Y[batch_indices]).type(torch.LongTensor)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)


def removeCol(listOFlist,colNo):
	for i in range(0,len(listOFlist)):
		del listOFlist[i][colNo]

def class_numbers(y):
	for i in range(0,len(y)):
		if y[0]==1:
			return 0
		elif y[1]==1:
			return 1
		elif y[2]==1:
			return 2
		else:
			return 3

def getlistclasses(y):
	k=[]
	for i in range(0,len(y)):
		k.append(class_numbers(y[i]))

	return k

def splitAllData(full):
	train_len=int(0.7*len(listOfAllTrain))
	val_len=len(listOfAllTrain)-train_len
	full=np.asarray(full)
	return full[0:train_len].tolist(),full[train_len:]


def validate(model,x,y):	
	x,y=get_batch2(x,getlistclasses(y),len(y))
	yp=model(x)
	y=y.data.numpy()
	ypn=np.asarray(getlistclasses(yp.data.numpy()))
	return np.mean(y==ypn)
	




listOfAllTrain = []
classes=3

with open('Leaderboard_Dataset.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
        	listOfAllTrain.append(row)
            
listOfAllTrain.remove(listOfAllTrain[0])

#remove charge
for i in range(0,len(listOfAllTrain)):
	listOfAllTrain[i].remove('Charge      ')

col12=[]

for i in range(0,len(listOfAllTrain)):
	col12.append(listOfAllTrain[i][12])

uniqueCol12 = list(set(col12))

#change unknown to zero later
fincol12=[]

for j in range(len(col12)):
	for i in range(0,len(uniqueCol12)):	
		if uniqueCol12[i]==col12[j]:
			fincol12.append(i)

#replace col12 with numbers
for i in range(0,len(listOfAllTrain)):
	listOfAllTrain[i][11]=fincol12[i]

#make every element as float

for i in range(0,len(listOfAllTrain)):
	for j in range(0,len(listOfAllTrain[0])):
		listOfAllTrain[i][j]=float(listOfAllTrain[i][j])

removeCol(listOfAllTrain,0)

x=Variable(torch.FloatTensor(listOfAllTrain))
model=torch.load('/home/ajwahir/amex/checkpoint.pth.tar')
y=model(x)