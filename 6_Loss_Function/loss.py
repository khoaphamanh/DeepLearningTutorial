import torch
from torch import nn
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
from math import log

# 100 covid-19 patients: 75 negative: 0, 25 positive: 1
print("Binary Cross Entropy Loss ")

# 100 covid-19 patients: 75 negative: 0, 25 positive: 1
print("Binary Cross Entropy Loss ")

pos_weight = torch.tensor(75 / 25)
y_logit = torch.tensor([-0.5, 0.8, 5, -4]).float()
y_prob = torch.sigmoid(y_logit)
y_true = torch.tensor([0, 1, 0, 0]).float()

loss_balance = nn.BCEWithLogitsLoss()
loss_unbalance = nn.BCEWithLogitsLoss(weight=pos_weight)

cost_balance = loss_balance(y_logit, y_true)
print("cost_balance:", cost_balance)
cost_unbalance = loss_unbalance(y_logit, y_true)
print("cost_unbalance:", cost_unbalance)

print(cost_unbalance == cost_balance * pos_weight)

# 6 samples, 3 class: {0: 1 element, 1: 3 element, 2: 2element}
print("Cross Entropy Loss: ")
torch.manual_seed(1998)

y_logit = torch.randn(6, 3)
y_prob = nn.Softmax(dim=-1)(y_logit)
y_true = torch.tensor([0, 1, 1, 2, 1, 2])
print("y_prob:", y_prob)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_true), y=y_true.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("class weights:", class_weights)  # [sample / class / sample in class]
print(class_weights == torch.tensor([6 / 3 / 1, 6 / 3 / 3, 6 / 3 / 2]))

loss = nn.CrossEntropyLoss(weight=class_weights)
cost = loss(y_logit, y_true)
print("loss unbalance class:", cost)
print(
    (
        -2.0000 * log(0.3078)
        - 0.6667 * log(0.0983)
        - 0.6667 * log(0.3313)
        - 1.0000 * log(0.6218)
        - 0.6667 * log(0.7574)
        - 1.0000 * log(0.7326)
    )
    / 6
)
print()

# ignore_index = 0
loss = nn.CrossEntropyLoss(
    weight=class_weights, ignore_index=0
)  # new_class_weight[0, 5/2/3, 5/2/2 ]
cost = loss(y_logit, y_true)
print(y_prob)
print("loss unbalance class ignore index 0:", cost)
print(
    (
        -0 * log(0.3078)
        - 0.833333 * log(0.0983)
        - 0.833333 * log(0.3313)
        - 1.25 * log(0.6218)
        - 0.833333 * log(0.7574)
        - 1.25 * log(0.7326)
    )
    / 5
)


def MSE(x):
    return (x - 1) ** 2


def BCE(x):
    return -np.log(x)


def gradient_MSE(x):
    return 2 * (x - 1)


def gradient_BCE(x):
    return -1 / x


def tangente_MSE(x, x0, y0):
    return gradient_MSE(x0) * (x - x0) + y0


def tangente_BCE(x, x0, y0):
    return gradient_BCE(x0) * (x - x0) + y0


x = np.linspace(0, 1, 100)
x1 = np.linspace(-0.01, 0.03, 100)
x2 = np.linspace(-0.3, 0.5, 100)
x_slope = 0.01

plt.plot(x, MSE(x), color="red", label="MSE")
plt.plot(x2, tangente_MSE(x2, x_slope, MSE(x_slope)), color="brown", label="slope MSE")
plt.scatter(x_slope, MSE(x_slope), c="red", s=20)

plt.plot(x, BCE(x), color="blue", label="BCE")
plt.plot(x1, tangente_BCE(x1, x_slope, BCE(x_slope)), color="black", label="slope BCE")
plt.scatter(x_slope, BCE(x_slope), c="blue", s=20)

plt.legend(loc=1)
plt.xlabel("ơ")
plt.ylabel("y_true")
plt.grid()
plt.show()
