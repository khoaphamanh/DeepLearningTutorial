import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split

#helper function:
def y_noise (y):
    y_new = list()
    for element in list(np.array(y).reshape(len(y),)):
        noise = random.randint(-10, 10) * random.random()
        y_new.append(element + noise)

    return torch.tensor(np.array(y_new).reshape(len(y),1))

def draw (x,y,label):
    plt.scatter(x,y,s=1,label = label, c = 'red')
    plt.legend(loc = 1)

def draw_line (x,y,label):
    plt.plot(x,y,label = label)
    plt.legend(loc = 1)

#model class:
class LinearRegression(nn.Module):
    def __init__(self,input,hidden,output):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input,out_features = hidden).requires_grad_(True)
        self.layer2 = nn.Linear(in_features=hidden,out_features = output).requires_grad_(True)
    def forward (self, x:torch.tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#model Logistic Regression:
class LogisticRegression(nn.Module):
    def __init__(self,input,hidden,output):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input,out_features = hidden).requires_grad_(True)
        self.layer2 = nn.Linear(in_features=hidden,out_features = output).requires_grad_(True)
        self.relu = nn.ReLU()
    def forward (self, x:torch.tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

#seed and constant
seed = 1998
torch.manual_seed(seed)
random.seed(seed)
LEARNING_RATE = 0.1
INPUT = 1
HIDDEN = 64
OUTPUT = 1
EPOCH = 101

#function y = 2x + 5
X = torch.arange(-5,5,0.1)
X = X.reshape(len(X),1)

y = 2*X**2 + 5*X +7
y = y_noise(y)

#split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,shuffle=True,random_state=seed)

#model
model1 = LinearRegression(input = INPUT,hidden=HIDDEN,output=OUTPUT)
model2 = LogisticRegression(input = INPUT,hidden=HIDDEN,output=OUTPUT)

#optimizer and loss:
optimizer = torch.optim.Adam(params=model1.parameters(),lr=LEARNING_RATE)
loss = nn.MSELoss()

#train test loop:
epoch_list = list()
loss_train_list = list()
loss_test_list = list()

for epoch in range(EPOCH):

    #train mode
    model1.train()

    #forward pass:
    y_pred_train = model1(X_train)

    #calculate the loss:
    loss_train = loss(y_pred_train,y_train.float())

    #zero grad:
    optimizer.zero_grad()

    #backpropagation
    loss_train.backward()

    #update parameter
    optimizer.step()

    #tracking:
    epoch_list.append(epoch)
    loss_train_list.append(loss_train.detach().numpy())

    with torch.inference_mode():

        # forward pass:
        y_pred_test = model1(X_test)

        # calculate the loss:
        loss_test = loss(y_pred_test, y_test.float())

        #tracking:
        loss_test_list.append(loss_test.detach().numpy())

    print('epoch = {}'.format(epoch))
    print('loss train = {}'.format(loss_train))
    print('loss test = {}'.format(loss_test))
    print()

#Logistic LRegression
epoch_list1 = list()
loss_train_list1 = list()
loss_test_list1 = list()
optimizer = torch.optim.Adam(params=model2.parameters(),lr=LEARNING_RATE)

for epoch in range(EPOCH):

    #train mode
    model2.train()

    #forward pass:
    y_pred_train = model2(X_train)

    #calculate the loss:
    loss_train = loss(y_pred_train,y_train.float())

    #zero grad:
    optimizer.zero_grad()

    #backpropagation
    loss_train.backward()

    #update parameter
    optimizer.step()

    #tracking:
    epoch_list1.append(epoch)
    loss_train_list1.append(loss_train.detach().numpy())

    with torch.inference_mode():

        # forward pass:
        y_pred_test = model2(X_test)

        # calculate the loss:
        loss_test = loss(y_pred_test, y_test.float())

        #tracking:
        loss_test_list1.append(loss_test.detach().numpy())

    print('epoch = {}'.format(epoch))
    print('loss train = {}'.format(loss_train))
    print('loss test = {}'.format(loss_test))
    print()


plt.figure(figsize=(7,10))
plt.subplot(2,1,1)
draw(X,y,'data')
draw_line(X,model1(X).detach().numpy(),'linear')
draw_line(X,model2(X).detach().numpy(),'Non linear')
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2,1,2)
draw_line(epoch_list,loss_train_list,'loss train linear')
draw_line(epoch_list,loss_test_list,'loss test linear')
draw_line(epoch_list,loss_train_list1,'loss train non linear')
draw_line(epoch_list,loss_test_list1,'loss test non linear')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()







