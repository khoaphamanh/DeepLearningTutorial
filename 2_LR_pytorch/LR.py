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
        noise = random.randint(-5, 5) * random.random()
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

    def forward (self, x:torch.Tensor()):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#seed and constant
seed = 1998
torch.manual_seed(seed)
random.seed(seed)
LEARNING_RATE = 0.01
INPUT = 1
HIDDEN = 2
OUTPUT = 1
EPOCH = 100

#function y = 2x + 5
X = torch.arange(-5,5,0.1)
X = X.reshape(len(X),1)

y = 2*X + 5
y = y_noise(y)

#split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,shuffle=False)

#model
model = LinearRegression(input = INPUT,hidden=HIDDEN,output=OUTPUT)

#optimizer and loss:
optimizer = torch.optim.SGD(params=model.parameters(),lr=LEARNING_RATE)
loss = nn.MSELoss()

#train test loop:
epoch_list = list()
loss_train_list = list()
loss_test_list = list()

for epoch in range(EPOCH):

    #train mode
    model.train()

    #forward pass:
    y_pred_train = model(X_train)

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
        y_pred_test = model(X_test)

        # calculate the loss:
        loss_test = loss(y_pred_test, y_test.float())

        #tracking:
        loss_test_list.append(loss_test.detach().numpy())

    print('epoch = {}'.format(epoch))
    print('loss train = {}'.format(loss_train))
    print('loss test = {}'.format(loss_test))
    print()

plt.figure(figsize=(7,10))
plt.subplot(2,1,1)
draw(X,y,'data')
draw_line(X,model(X).detach().numpy(),'predict')
plt.subplot(2,1,2)
draw_line(epoch_list,loss_train_list,'loss train')
draw_line(epoch_list,loss_test_list,'loss test')
plt.show()







