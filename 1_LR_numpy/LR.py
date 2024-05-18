import matplotlib.pyplot as plt
import numpy as np
import  random

def y_noise(y):
    random.seed(1998)
    y_new = list()
    for element in y:
        noise = random.random()
        faktor = random.randint(-5,5)
        y_new.append(element+faktor*noise)
    return np.array(y_new)

def draw(X,y,label):
    plt.scatter(X, y, s=2, label = label)
    plt.legend(loc = 1)

#forward pass:
def forward (x):
    y_pred = x*w +b
    return y_pred

#loss: MSE = 1/N * (x*w+b -y_true)**2
def MSE(x,y_true,weight,bias):
    MSE = ((weight*x +bias) - y_true)**2
    sum = 0
    for i in MSE:
        sum = sum + i
    MSE = sum / len(y_true)
    return MSE

#gradient weight: dL / dw = 1/N . ( 2x(wx +b) -2 xy)
def gradient_weight (x,y_true, weight,bias):
    dw = 2*x*((weight*x +bias)-y_true)
    sum = 0
    for i in dw:
        sum = sum +i
    dw = sum / len(x)
    return dw

#gradient bias: dL / db = 1/N . ( 2(wx +b) -2 y)
def gradient_bias (x, y_true, weight, bias):
    db = 2*((weight*x +bias)- y_true )
    sum = 0
    for i in db:
        sum = sum +i
    db = sum / len(x)
    return db

#function: y = 2x + 5
X = np.arange(-5,5,0.1)
y = 2*X +5

#function with noise: y = 2x + 5
X = np.arange(-5,5,0.1)
y = y_noise(y)

#weights and bias
w = 0
b = 0

#loop
epochs = 500
lr =  0.01
loss_list = list()
epoch_list = list()

for epoch in range(epochs):

    #predict
    y_pred = forward(X)

    #calculate the loss
    loss = MSE(X,y,w,b)

    #gradient
    dw = gradient_weight(X,y,w,b)
    db = gradient_bias(X,y,w,b)

    #update
    w = w - lr*dw
    b = b - lr*db

    #check loss_update < loss
    loss_update = MSE(X,y,w,b)

    #tracking
    loss_list.append(loss)
    epoch_list.append(epoch)
    print('epoch = {}'.format(epoch))
    print('loss = {:.4f}'.format(loss))
    print('w = {:.4f}'.format(w))
    print('b = {:.4f}'.format(b))
    print()

plt.figure(figsize=(7,10))
plt.subplot(2,1,1)
draw(X,y,'train')
draw(X,forward(X), 'predict')
plt.xlabel('X')
plt.ylabel('y')
plt.subplot(2,1,2)
draw(epoch_list,loss_list, 'MSE')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()