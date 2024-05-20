import random
import torch
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary
from timeit import default_timer

#helper function:
#visualize the images
def visualize (row,column,data):
    plt.figure(figsize=(10,10))
    for i in range(row*column):
        #index = random.randint(0,len(data))
        plt.subplot(row,column,i+1)
        plt.imshow(data[i][0].squeeze(),cmap='gray')
        plt.title(data.classes[data[i][1]])
        plt.axis(False)

def accuracy_function(y_pred,y_true):
    check = torch.eq(y_pred,y_true)
    accuracy = sum(check).item() / len(y_true) * 100
    return accuracy

def time_function(start, end):
    time = end-start
    print('training time takes {} seconds'.format(time))

#creat a model:
class ComputerVision(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.convolution_layer =nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels=hidden,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden*13*13,out_features=output)
        )
    def forward(self,x:torch.tensor):
        x = self.convolution_layer(x) #shape (batch_size, hidden, (28-2) / 2, (28-2) / 2 )
        x = self.linear_layer(x)
        return x


#constant
SEED = 1998
torch.manual_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 5
COLOR_CHANEL = 1
INPUT = 1
HIDDEN = 2
OUTPUT = 10
LEARNING_RATE = 0.01
EPOCH = 10

#load data MNITS
traindata = datasets.MNIST(
    root='data',
    train=True,
    transform= transforms.ToTensor()
)

testdata = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor()
)

#turn data in Dataloader
train_dataloader = DataLoader(
    dataset=traindata,
    batch_size=BATCH_SIZE, #60000  1
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=testdata,
    batch_size=BATCH_SIZE, #10000  1
    shuffle=True
)

#creat a model:
model = ComputerVision(input=INPUT,hidden=HIDDEN,output=OUTPUT)

#optimizer and loss
optimizer = torch.optim.SGD(params=model.parameters(),lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()

start = default_timer()
#training testing process:
for epoch in range(EPOCH):
    loss_train = 0
    accuracy_train = 0
    print('epoch:',epoch)

    for batch_train,(X_train,y_train) in enumerate(train_dataloader):

        #training mode:
        model.train()

        #forward pass:
        y_pred_train_logit = model(X_train)
        y_pred_train_probability = torch.softmax(y_pred_train_logit,dim = 1)
        y_pred_train_label = y_pred_train_probability.argmax(dim = 1)

        #calculate the loss and accuracy
        loss_train_this_batch = loss(y_pred_train_logit,y_train)
        loss_train = loss_train + loss_train_this_batch
        accuracy_train_this_batch = accuracy_function(y_pred_train_label,y_train)
        accuracy_train = accuracy_train + accuracy_train_this_batch

        #gradient zero grad
        optimizer.zero_grad()

        #backpropagation
        loss_train_this_batch.backward()

        #updates parameters
        optimizer.step()

        #tracking
        #if batch_train % 3000 == 0:
        #    print('training {} images of {} images'.format(batch_train*BATCH_SIZE,len(traindata)))

    loss_train = loss_train.item() / len(train_dataloader)
    accuracy_train = accuracy_train / len(train_dataloader)

    loss_test = 0
    accuracy_test = 0
    model.eval()
    with torch.inference_mode():

        for batch_test,(X_test, y_test) in enumerate(test_dataloader):

            #forward pass:
            y_pred_test_logit = model(X_test)
            y_pred_test_probability = torch.softmax(y_pred_test_logit,dim = 1)
            y_pred_test_label = y_pred_test_probability.argmax(dim = 1)

            #calculate the loss and accuracy:
            loss_test_this_batch = loss(y_pred_test_logit, y_test)
            loss_test = loss_test + loss_test_this_batch
            accuracy_test_this_batch = accuracy_function(y_pred_test_label, y_test)
            accuracy_test = accuracy_test + accuracy_test_this_batch

            #tracking:
            #if batch_test % 1000 == 0:
            #    print('testing {} images of {} images'.format(batch_test * BATCH_SIZE, len(testdata)))

        loss_test = loss_test / len(test_dataloader)
        accuracy_test = accuracy_test / len(test_dataloader)

    print('loss train = {}, accuracy train = {}'.format(loss_train,accuracy_train))
    print('loss test = {}, accuracy test = {}'.format(loss_test,accuracy_test))
    print()

end = default_timer()
time_function(start,end)
print(model.state_dict())
visualize(4,4,traindata)
#summary(model,input_size=(32,1,28,28),col_names=["input_size", "output_size", "num_params", "trainable"])
plt.show()
