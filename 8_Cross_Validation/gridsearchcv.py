import torch
import numpy as np

from torch import nn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.datasets import load_iris

#helper function
def train_val_split (X_train_val,y_train_val,k,seed):
    k_fold = StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for train_index, val_index in k_fold.split(X_train_val,y_train_val):
        tmp_X_train = []
        tmp_y_train = []
        tmp_X_val = []
        tmp_y_val = []
        for index in train_index:
            tmp_X_train.append(X_train_val[index])
            tmp_y_train.append(y_train_val[index])
        for index in val_index:
            tmp_X_val.append(X_train_val[index])
            tmp_y_val.append(y_train_val[index])

        X_train.append(tmp_X_train)
        y_train.append(tmp_y_train)
        X_val.append(tmp_X_val)
        y_val.append(tmp_y_val)

    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

class NeuralNetwork(nn.Module):
    def __init__(self,input, hidden, n_layers, output):
        super().__init__()
        self.in_layer = nn.Linear(in_features=input,out_features=hidden)
        self.hidden_layer = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(in_features=hidden,out_features=hidden),
            nn.ReLU())
        self.out_layer = nn.Linear(in_features=hidden,out_features=output)
        self.n_layers = n_layers

    def forward(self,x:torch.tensor):
        x = self.in_layer(x)
        for i in range(self.n_layers):
            x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x

def hyperparameter_set(params_dict:dict):
    set = []
    for hidden in params_dict['hidden units']:
        for layers in params_dict['n_layers']:
            for lr in params_dict['learning_rate']:
                for epoch in params_dict['epoch']:
                    tmp = []
                    tmp.append(hidden)
                    tmp.append(layers)
                    tmp.append(lr)
                    tmp.append(epoch)
                    set.append(tmp)
    return np.array(set)

def accuracy_function (y_pred_label, y_true):
    check = torch.eq(y_pred_label,y_true)
    accuracy = sum(check) / len(y_true) * 100
    return accuracy

def train_test_model(X_train,X_val,y_train,y_val,hidden,n_layers,learning_rate,epoch):

    model = NeuralNetwork(input=INPUT,hidden=hidden,n_layers=n_layers,output=OUTPUT)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    #transform to tensor
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)

    for i in range(epoch):

        # training mode:
        model.train()

        # forward pass:
        y_pred_train_logit = model(X_train)
        y_pred_train_label = y_pred_train_logit.argmax(axis = -1)

        # calculate the loss and accuracy:
        loss_train = loss(y_pred_train_logit, y_train)
        accuracy_train = accuracy_function(y_pred_train_label, y_train)

        # zero grad the gradient
        optimizer.zero_grad()

        # backpropagation
        loss_train.backward()

        # update parameters
        optimizer.step()

        #testing:
        model.eval()
        with torch.inference_mode():
            # forward pass:
            y_pred_test_logit = model.forward(X_val)
            y_pred_test_label = y_pred_test_logit.argmax(axis=-1)

            # calculate the loss and accuracy
            loss_test = loss(y_pred_test_logit, y_val)
            accuracy_test = accuracy_function(y_pred_test_label, y_val)

    return float(accuracy_test)

def grid_search_CV(X_train,X_val,y_train,y_val,parameter_dict):
    torch.manual_seed(SEED)
    params_combi_all_result = []
    index = 0
    for params_combi in hyperparameter_set(params_dict=parameter_dict):
        result_this_params_combi = []
        #parameters:
        hidden = int(params_combi[0])
        n_layers = int(params_combi[1])
        learning_rate = params_combi[2]
        epoch = int(params_combi[3])
        print("training model number {} with hyperparameter: hidden = {}, n_layers = {}, learning_rate = {}, epoch = {}... "
              .format(index,hidden, n_layers, learning_rate, epoch))
        index = index + 1
        for fold_index in range(K):

            #data
            X_train_fold = X_train[fold_index]
            X_val_fold = X_val[fold_index]

            y_train_fold = y_train[fold_index]
            y_val_fold = y_val[fold_index]


            #training
            print("Fold: {}".format(fold_index+1),end=" ")
            result_this_fold = train_test_model(X_train_fold,X_val_fold,y_train_fold,y_val_fold,hidden,n_layers,learning_rate,epoch)
            print('accuracy: {:.2f}\n '.format(result_this_fold))
            result_this_params_combi.append(result_this_fold)

        #calculate the mean test accuracy:
        params_combi_all_result.append(np.mean(result_this_params_combi))

    return params_combi_all_result

def final_test_model (X_train_val,X_test,y_train_val,y_test):
    torch.manual_seed(SEED)
    hidden = int(tracking[0])
    n_layers = int(tracking[1])
    learning_rate = tracking[2]
    epoch = int(tracking[3])

    model = NeuralNetwork(input=INPUT,hidden=hidden,n_layers=n_layers,output=OUTPUT)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    #transform to tensor
    X_train_val = torch.tensor(X_train_val).float()
    X_test = torch.tensor(X_test).float()

    y_train_val = torch.tensor(y_train_val)
    y_test = torch.tensor(y_test)

    for i in range(epoch):

        # training mode:
        model.train()

        # forward pass:
        y_pred_train_logit = model(X_train_val)
        y_pred_train_label = y_pred_train_logit.argmax(axis = -1)

        # calculate the loss and accuracy:
        loss_train = loss(y_pred_train_logit, y_train_val)
        accuracy_train = accuracy_function(y_pred_train_label, y_train_val)

        # zero grad the gradient
        optimizer.zero_grad()

        # backpropagation
        loss_train.backward()

        # update parameters
        optimizer.step()

        #testing:
        model.eval()
        with torch.inference_mode():
            # forward pass:
            y_pred_test_logit = model.forward(X_test)
            y_pred_test_probability = nn.Softmax(dim=-1)(y_pred_test_logit)
            y_pred_test_label = y_pred_test_probability.argmax(axis=-1)

            # calculate the loss and accuracy
            loss_test = loss(y_pred_test_probability, y_test)
            accuracy_test = accuracy_function(y_pred_test_label, y_test)

    return float(accuracy_train),float(accuracy_test)

#constant:
SEED = 1998
torch.manual_seed(SEED)
INPUT = 4
OUTPUT = 3
K = 10

#load data
X = load_iris()["data"]
y = load_iris()["target"]

#split data train_val test
X_train_val,X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED,shuffle=True,stratify=y)

#split data train val
X_train, X_val, y_train, y_val = train_val_split(X_train_val,y_train_val,k=K,seed=SEED)

#hyperparameter
hyperparameter = {
    'hidden units':[8,16,32,64],
    'n_layers':[1,2,3],
    'learning_rate': [0.05,0.1,0.2],
    'epoch':[20,50,100]
}

#hyperparameter_combination
hyperparameter_combination = hyperparameter_set(hyperparameter)

#hyperparameters tuning with gridsearch cross validation
test = grid_search_CV(X_train,X_val,y_train,y_val,hyperparameter)
print("result of model with all hyperparameters combination ",test)
print("the best model is model number {} with mean accuracy test {:.2f}".format(test.index(max(test)),max(test)))
tracking = hyperparameter_combination[test.index(max(test))]
print("the best model is with hyperparameters: hidden = {}, n_layers = {}, learning_rate = {}, epoch = {} ".format(int(tracking[0]), int(tracking[1]),tracking[2],int(tracking[3])))

#final testing
final_testing = final_test_model(X_train_val,X_test,y_train_val,y_test)
print("final testing is: accuracy train = {:.2f}, accuracy test = {:2f}".format(final_testing[0],final_testing[1]))