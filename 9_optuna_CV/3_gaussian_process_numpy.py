import matplotlib.pyplot as plt
import numpy as np
import math

#load data: f(x) = (x - 2)^2
X = np.arange(-1,6,dtype=float)
y = (X-2)**2
X = X.reshape(len(X),1)
X_new = np.array([[1.5],
                  [4.5]])
print('X: ',X)
print('y: ',y)
print('X_new =',X_new)
print()

#hyperparameters:
l = 1
sigma = 1
noise = 0

#square exponential kernel:
def square_exponential_kernel(xi,xj,l=l,sigma=sigma):
    #l: length scale, o: sigma variance
    return sigma*math.exp((-abs(xi-xj)**2) / 2*l**2)

#covariance matrix
def kernel_matrix(Xi,Xj,l=l,sigma=sigma):
    cov_matrix = []
    for xi in Xj:
        tmp = []
        for xj in Xi:
            element = square_exponential_kernel(xi,xj,l=l,sigma=sigma)
            tmp.append(element)
        cov_matrix.append(tmp)
    return np.array(cov_matrix)


#mean and covariance matrix prediction
def gaussian_mean_predict(X,X_new,y,l=l,sigma = sigma,noise=noise):
    K1 = kernel_matrix(X, X,l,sigma) + noise*np.eye(len(X))
    K2 = kernel_matrix(X, X_new,l,sigma)
    mean_pred = np.dot(np.dot(K2, np.linalg.inv(K1)), y)
    return mean_pred

def gaussian_covariance_prediction(X,X_new,l=l,sigma = sigma, noise = noise):
    K1 = kernel_matrix(X, X,l,sigma) + noise*np.eye(len(X))
    K2 = kernel_matrix(X, X_new,l,sigma)
    K3 = kernel_matrix(X_new, X_new)
    covariance_matrix_predict = K3 - np.dot(np.dot(K2, np.linalg.inv(K1)), K2.T)
    variance = np.diag(covariance_matrix_predict)
    standard_distribution = np.sqrt(variance)
    return standard_distribution

K1 = kernel_matrix(X, X) + noise*np.eye(len(X))
K2 = kernel_matrix(X, X_new)
K3 = kernel_matrix(X_new,X_new)

#mean:
mean_pred = gaussian_mean_predict(X,X_new,y)
print('mean: ',mean_pred)
print("true: [{} {}] ".format((1.5-2)**2,(4.5-2)**2))
#covarian:
covariance_matrix_predict = K3 - np.dot(np.dot(K2,np.linalg.inv(K1)),K2.T)
standard_distribution = gaussian_covariance_prediction(X,X_new)

print('covariance matrix predict: ',covariance_matrix_predict)
print('covariance: ',standard_distribution)

#visualizing
x = np.linspace(-1.5,5.5,1000)
y_pred = []
y_std = []
for i in x:
    X_new = np.array([i])
    y_pred.append(gaussian_mean_predict(X,X_new,y))
    y_std.append(gaussian_covariance_prediction(X,X_new))

y_pred,y_std = np.array(y_pred).reshape(len(x)), np.array(y_std).reshape(len(x))

plt.plot(x,(x-2)**2,'red',label = 'y_true')
plt.scatter(X,y,c='black',label = 'X')
plt.plot(x,y_pred,'green',label = 'y_pred')
for o in range(1,4):
    if o == 1:
        label = 'µ ± ơ'
        alpha = 0.6
    elif o ==2:
        label = 'µ ± 2ơ'
        alpha = 0.4
    else:
        label = 'µ ± 3ơ'
        alpha = 0.1
    plt.fill(np.hstack([x, x[::-1]]),
             np.hstack([y_pred - o*y_std,
                        (y_pred + o*y_std)[::-1]]),
             alpha=alpha, fc="b",label = label)
plt.legend(loc=1)
plt.grid()
plt.show()








