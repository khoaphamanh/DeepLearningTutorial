import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, confusion_matrix


# helper function:
def draw_3D(X, y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    ax = plt.axes(projection="3d")
    ax.scatter(x1, x2, x3, c=y, s=1)


def draw_2D(X, y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.scatter(x1, x2, c=y, s=1)


def accuracy_function(y_pred_label, y_true):
    check = torch.eq(y_pred_label, y_true)
    accuracy = sum(check).item() / len(y_true) * 100
    return accuracy


def plot_loss(epoch_list, loss_train_list, loss_test_list):
    plt.plot(epoch_list, loss_train_list, label="loss train")
    plt.plot(epoch_list, loss_test_list, label="loss test")
    plt.legend(loc=1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("tracking loss")


def plot_accuracy(epoch_list, accuracy_train_list, accuracy_test_list):
    plt.plot(epoch_list, accuracy_train_list, label="accuracy train")
    plt.plot(epoch_list, accuracy_test_list, label="accuracy test")
    plt.legend(loc=4)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("tracking accuracy")


# creat a model
class MultiClassClassification(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=output),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# constant
N_SAMPLES = 10000
N_FEATURES = 3
CENTERS = 5
CLUSTER_STD = 2
CENTER_BOX = (-10, 10)
SEED = 40
torch.manual_seed(SEED)
np.random.seed(SEED)
INPUT = 3
HIDDEN = 16
OUTPUT = 5
LEARNING_RATE = 0.1
EPOCH = 31
WEIGHTS = [0.01, 0.01, 0.04, 0.04, 0.8]

# load data:
X, y = make_blobs(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    centers=CENTERS,
    cluster_std=CLUSTER_STD,
    center_box=CENTER_BOX,
    random_state=SEED,
)


def create_unbalance_datasets(X: np.array, y: np.array, weights: list):
    n_samples = len(X)
    unique_indices = {value: np.where(y == value)[0] for value in y}

    n_labels = len(unique_indices.keys())
    unbalance_n_samples = {
        idx: int((w * n_samples) / n_labels) for idx, w in enumerate(weights)
    }

    dict_indices_to_take = {
        k: np.random.choice(unique_indices[k], unbalance_n_samples[k], replace=False)
        for k in unique_indices.keys()
    }

    indices_to_take = np.concatenate(
        [np.array(dict_indices_to_take[key]) for key in dict_indices_to_take.keys()]
    )

    return X[indices_to_take], y[indices_to_take]


X, y = create_unbalance_datasets(X, y, WEIGHTS)
unique = np.unique(y, return_counts=True)
print("unique:", unique)

X = torch.tensor(X).float()
y = torch.tensor(y).long()

# split in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=SEED, stratify=y
)
unique_train = torch.unique(y_train, return_counts=True)
print("unique train:", unique_train)
unique_test = torch.unique(y_test, return_counts=True)
print("unique test:", unique_test)

# creat a model:
model = MultiClassClassification(INPUT, HIDDEN, OUTPUT)

# class weitghs
class_weights = class_weight.compute_class_weight(
    "balanced", classes=unique_train[0].numpy(), y=y_train.numpy()
)
print("class_weights:", class_weights)

# loss and optimizer:
loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

# training, testing loop
loss_train_list = []
loss_test_list = []
accuracy_train_list = []
accuracy_test_list = []
epoch_list = []
for epoch in range(EPOCH):

    # training mode:
    model.train()

    # forward pass:
    y_pred_train_logit = model.forward(X_train)
    y_pred_train_probability = nn.Softmax(dim=-1)(y_pred_train_logit)
    y_pred_train_label = y_pred_train_probability.argmax(axis=1)

    # calculate the loss and accuracy
    loss_train = loss(y_pred_train_logit, y_train)
    accuracy_train = accuracy_function(y_pred_train_label, y_train)

    # gradient zero grad:
    optimizer.zero_grad()

    # backpropagation
    loss_train.backward()

    # update parameters:
    optimizer.step()

    # testing:
    model.eval()
    with torch.inference_mode():

        # forward pass:
        y_pred_test_logit = model.forward(X_test)
        y_pred_test_probability = nn.Softmax(dim=-1)(y_pred_test_logit)
        y_pred_test_label = y_pred_test_probability.argmax(axis=-1)

        # calculate the loss and accuracy
        loss_test = loss(y_pred_test_logit, y_test)
        accuracy_test = accuracy_function(y_pred_test_label, y_test)

    # plot loss accuracy
    epoch_list.append(epoch)
    loss_train_list.append(loss_train.detach().numpy())
    loss_test_list.append(loss_test.detach().numpy())
    accuracy_train_list.append(accuracy_train)
    accuracy_test_list.append(accuracy_test)

    # tracking:
    if epoch % 10 == 0:
        print("epoch = {}".format(epoch))
        print("loss train = {}, accuracy train = {}".format(loss_train, accuracy_train))
        print("loss train = {}, accuracy train = {}".format(loss_test, accuracy_test))
        print()

# Compute predictions
y_pred_train_label = model(X_train).argmax(axis=1).detach().numpy()
y_pred_test_label = model(X_test).argmax(axis=1).detach().numpy()

# Compute metrics
f1 = f1_score(y_test, y_pred_test_label, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_test_label)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(4, 4))
draw_3D(X, y)
plt.figure(figsize=(4, 4))
draw_2D(X, y)
plt.figure(figsize=(8, 10))
plt.subplot(2, 1, 1)
plot_loss(epoch_list, loss_train_list, loss_test_list)
plt.subplot(2, 1, 2)
plot_accuracy(epoch_list, accuracy_train_list, accuracy_test_list)
# plt.show()
