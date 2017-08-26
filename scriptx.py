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

with open('Training_Dataset.csv') as csvfile:
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

yg=[]
for i in range(0,len(listOfAllTrain)):
	yg.append(listOfAllTrain[i][len(listOfAllTrain[0])-classes:len(listOfAllTrain[0])])

for i in range(0,6):
	removeCol(listOfAllTrain,len(listOfAllTrain[0])-1)


#NN part

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation uses the nn package from PyTorch to build the network.
Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.
"""

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, H3, D_out = 64, len(listOfAllTrain[0]), 100, 100, 200, classes+1
train,val = splitAllData(listOfAllTrain)
train_y,val_y = splitAllData(yg)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
# x = Variable(torch.FloatTensor(listOfAllTrain))
# y = Variable(torch.FloatTensor(yg), requires_grad=False)

# Use the nn package to define our model and loss fully-connectedtion.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H1),
          torch.nn.Linear(H1, H2),
          torch.nn.ReLU(),
          torch.nn.Linear(H2, H3),
          torch.nn.ReLU(),
          torch.nn.Linear(H3, D_out),
          # torch.nn.ReLU(),
          torch.nn.Softmax(),
        )
# loss_fn = torch.nn.MSELoss(size_average=False)
loss_fn = torch.nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model.
  model.train()
  x,y=get_batch2(train,getlistclasses(train_y),N)

  y_pred = model(x)

  # Compute and print loss.
  loss = loss_fn(y_pred, y)
  # print y_pred	
  print(t, loss.data[0])
  
  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the variables it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()
  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()


model.eval()
print validate(model,val,val_y)


torch.save(model,'/home/ajwahir/amex/checkpoint.pth.tar')


x,y=get_batch2(val,getlistclasses(val_y),len(val_y))
yp=model(x)
y=y.data.numpy()
ypn=np.asarray(getlistclasses(yp.data.numpy()))

print ypn