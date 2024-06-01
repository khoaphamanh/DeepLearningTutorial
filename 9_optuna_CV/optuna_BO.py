import torch
import numpy as np
import optuna
import time
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.datasets import load_iris
from torch import nn

#helper function
def accuracy_function (y_pred_label, y_true):
    check = torch.eq(y_pred_label,y_true)
    accuracy = sum(check) / len(y_true) * 100
    return accuracy

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

#create model
class NeuralNetWork(nn.Module):
    def __init__(self,trial:optuna.trial.Trial,input,output):
        super().__init__()
        #define hyperparameters:
        self.hidden = trial.suggest_int('hidden',low=1,high=16,step=1)
        self.n_layers = trial.suggest_int('n_layers',low=0,high=5,step=1)

        #define layers
        self.in_layer = nn.Linear(in_features=input,out_features=self.hidden)
        self.hidden_layer = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(in_features=self.hidden,out_features=self.hidden),
            nn.ReLU())
        self.out_layer = nn.Linear(in_features=self.hidden,out_features=output)

    def forward(self,x:torch.tensor):
        x = self.in_layer(x)
        for i in range(self.n_layers):
            x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x

#training model with train and val set
def train_val_model(trial:optuna.trial.Trial,X_train,X_val,y_train,y_val):
    #create model:
    model = NeuralNetWork(trial, INPUT, OUTPUT)
    torch.manual_seed(SEED)

    #define hyperparameters:
    learning_rate = trial.suggest_float(name='learning_rate',low=0.001,high=1,log=True)
    epoch = trial.suggest_int(name='epoch',low = 10,high=300,step=1)
    optimizer_categorical = trial.suggest_categorical(name='optimizer',choices=['Adam','SGD'])

    #loss and optimizer:
    loss = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim,optimizer_categorical)(params = model.parameters(),lr = learning_rate)

    #transform to tensor
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)

    #training:
    for i in range(epoch):
        # training mode:
        model.train()

        # forward pass:
        y_pred_train_logit = model(X_train)
        y_pred_train_label = y_pred_train_logit.argmax(axis=-1)

        # calculate the loss and accuracy:
        loss_train = loss(y_pred_train_logit, y_train)
        accuracy_train = accuracy_function(y_pred_train_label, y_train)

        # zero grad the gradient
        optimizer.zero_grad()

        # backpropagation
        loss_train.backward()

        # update parameters
        optimizer.step()

        # testing:
        model.eval()
        with torch.inference_mode():
            # forward pass:
            y_pred_test_logit = model.forward(X_val)
            y_pred_test_label = y_pred_test_logit.argmax(axis=-1)

            # calculate the loss and accuracy
            loss_test = loss(y_pred_test_logit, y_val)
            accuracy_val = accuracy_function(y_pred_test_label, y_val)

    return float(accuracy_val)

def objective(trial:optuna.trial.Trial):
    fold_score = []
    for fold in range(K):
        #choose feature and label
        X_train_fold = X_train[fold]
        X_val_fold = X_val[fold]
        y_train_fold = y_train[fold]
        y_val_fold = y_val[fold]

        #training:
        result_this_fold = train_val_model(trial,X_train_fold,X_val_fold,y_train_fold,y_val_fold)
        fold_score.append(result_this_fold)

        #tracking model information:
        if fold == 0:
            print("trial: {}".format(trial.number),end=" ")
            for key, value in trial.params.items():
                print("{}:{},".format(key, value),end=" ")
            print("\n")
        print('Fold: {} accuracy: {:.2f}\n '.format(fold+1,result_this_fold))

    score = np.mean(fold_score)
    return score

#finaltesting with best model:
class NeuralNetworkFinal(nn.Module):
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

#final testing
def final_test_model(best_params:dict,X_train_val,X_test,y_train_val,y_test):
    torch.manual_seed(SEED)
    hidden = best_params["hidden"]
    n_layers = best_params["n_layers"]
    learning_rate = best_params["learning_rate"]
    epoch = best_params["epoch"]
    optimizer = best_params["optimizer"]

    model = NeuralNetworkFinal(input=INPUT, hidden=hidden, n_layers=n_layers, output=OUTPUT)
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
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

#Constante
SEED = 1998
torch.manual_seed(SEED)
INPUT = 4
OUTPUT = 3
K = 10

#load data
data = load_iris()
X = data["data"]
y = data["target"]

#split data in train_val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,train_size=0.8,random_state=SEED,shuffle=True,stratify=y)

#split data in train, val
X_train, X_val, y_train, y_val = train_val_split(X_train_val,y_train_val,k=K,seed=SEED)

#Hyperparameters tuning with Bayes Optimizer
study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=11, timeout=600)

time.sleep(1)
print("\nStudy statistics: ")
print("finished trials: ", len(study.trials))

trial = study.best_trial
print("Best trial:",trial.number)

print("Best accuracy:", trial.value)

print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#final testing
best_params = trial.params
final_testing = final_test_model(best_params,X_train_val,X_test,y_train_val,y_test)
print("final testing is: accuracy train = {:.2f}, accuracy test = {:2f}".format(final_testing[0],final_testing[1]))