import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
#from torchinfo import summary

#helper function
def draw_circle(X,y):
    plt.scatter(x = X[:,0],y= X[:,1],c=y,s=1)
    plt.xlabel('X1')
    plt.ylabel('X2')

def accuracy_function (y_pred_label, y_true):
    check = torch.eq(y_pred_label,y_true)
    accuracy = sum(check) / len(y_true) * 100
    return accuracy

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    x0_min, x0_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    x1_min, x1_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1

    #points of grid
    xx,yy = np.meshgrid(np.linspace(x0_min,x0_max,100),np.linspace(x1_min,x1_max,100))

    #get "new" feature
    x_feature = torch.tensor(np.stack((xx.ravel(),yy.ravel()),axis=-1)).float()

    #test "new" features
    with torch.inference_mode():
        y_logit = model(x_feature).reshape(1,len(x_feature))

    #Binary or multiclass classification?
    if len(torch.unique(y)) == 2:
        y_probabilty = torch.sigmoid(y_logit)
        y_label = torch.round(y_probabilty)
    else:
        y_probabilty = torch.softmax(y_logit).argmax(dim = 1)
        y_label = torch.round(y_probabilty)

    y_label = y_label.reshape(xx.shape)
    plt.contourf(xx, yy, y_label, alpha=0.3)
    plt.scatter(x = X[:,0],y= X[:,1],c=y,s=3)
    plt.xlabel('X1')
    plt.ylabel('X2')

def plot_loss(epoch_list,loss_train_list,loss_test_list):
    plt.plot(epoch_list,loss_train_list,label = 'loss train')
    plt.plot(epoch_list, loss_test_list, label='loss test')
    plt.legend(loc = 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('tracking loss')

def plot_accuracy(epoch_list,accuracy_train_list,accuracy_test_list):
    plt.plot(epoch_list, accuracy_train_list, label='accuracy train')
    plt.plot(epoch_list, accuracy_test_list, label='accuracy test')
    plt.legend(loc=4)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('tracking accuracy')

class BinaryClassification(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input,out_features=hidden)
        self.acc = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden,out_features=output)

        # Initialize weights and biases with custom values
    #     self.initialize_parameters()
           
    # def initialize_parameters(self):
    #     # Initialize weights
    #     torch.nn.init.constant_(self.layer1.weight, 0.5)
    #     torch.nn.init.constant_(self.layer2.weight, 0.5)

    #     # Initialize biases
    #     torch.nn.init.constant_(self.layer1.bias, 0.2)
    #     torch.nn.init.constant_(self.layer2.bias, 0.2)

    def forward(self,x:torch.tensor):
        x = self.layer1(x)
        x = self.acc(x)
        x = self.layer2(x)
        return x

#constant and seed
seed = 1998
torch.manual_seed(seed)
random.seed(seed)
N_SAMPLES = 1000
NOISE = 0.035
INPUT = 2
HIDDEN = 64
OUTPUT = 1
LEARNING_RATE = 0.1
EPOCH = 301

#load data:
X,y = make_circles(n_samples=N_SAMPLES,shuffle=True,noise=NOISE,random_state=seed,factor=0.8)
X = torch.tensor(X).float()
y = torch.tensor(y).float()

#split data:
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)

#creat a model:
model = BinaryClassification(input=INPUT,hidden=HIDDEN,output=OUTPUT)

#optimizer and loss
optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
loss = nn.BCEWithLogitsLoss()
#loss = nn.BCELoss

#traing testing process
epoch_list = []
loss_train_list = []
loss_test_list = []
accuracy_train_list = []
accuracy_test_list = []

for epoch in range(EPOCH):

    #training mode:
    model.train()

    #forward pass:
    y_pred_train_logit = model(X_train).ravel()
    y_pred_train_probability = torch.sigmoid(y_pred_train_logit)
    y_pred_train_label = torch.round(y_pred_train_probability)

    #calculate the loss and accuracy:
    loss_train = loss(y_pred_train_logit,y_train)
    accuracy_train = accuracy_function(y_pred_train_label,y_train)

    #zero grad the gradient
    optimizer.zero_grad()

    #backpropagation
    loss_train.backward()

    #update parameters
    optimizer.step()

    # testing
    model.eval()
    with torch.inference_mode():

        # forward pass test
        y_pred_test_logit = model(X_test).ravel()
        y_pred_test_probability = torch.sigmoid(y_pred_test_logit)
        y_pred_test_label = torch.round(y_pred_test_probability)  # reshape 2D into 1D

        # calculate the loss test:
        loss_test = loss(y_pred_test_logit, y_test)

        # calculate accuracy test:
        accuracy_test = accuracy_function(y_pred_test_label,y_test)

    # tracking loss train test accuracy:
    if epoch % 10 == 0:
        print('epoch = {}'.format(epoch))
        print('loss train = {}, accuracy train = {}'.format(loss_train, accuracy_train))
        print('loss test = {}, accuracy test = {}'.format(loss_test, accuracy_test))
        print()
                                                                       
    #append
    epoch_list.append(epoch)
    loss_train_list.append(loss_train.detach().numpy())
    loss_test_list.append(loss_test.detach().numpy())
    accuracy_train_list.append(accuracy_train)
    accuracy_test_list.append(accuracy_test)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
draw_circle(X,y)
plt.subplot(2,2,2)
plot_loss(epoch_list,loss_train_list,loss_test_list)
plt.subplot(2,2,3)
plot_decision_boundary(model,X,y)
plt.subplot(2,2,4)
plot_accuracy(epoch_list,accuracy_train_list,accuracy_test_list)
plt.show()


