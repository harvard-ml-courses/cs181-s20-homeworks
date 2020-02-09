import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']
X = X_df.values
y = y_df.values

ln = len(y)

def maha2(a):
    yn = y_df.copy()
    total = 0
    for l in range(ln):
        bot = 0
        top = 0
        for n in range(ln):
            if l == n:
                continue
            bot += math.exp(-(a[0][0]*((X[n][0]-X[l][0])**2))-(2*a[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(a[1][1]*((X[n][1]-X[l][1])**2)))
            top += math.exp(-(a[0][0]*((X[n][0]-X[l][0])**2))-(2*a[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(a[1][1]*((X[n][1]-X[l][1])**2)))*y[n]
        total = (top/bot)
        yn[l] = total
        
    return yn

def kernel_loss(a):
    total = 0
    for l in range(ln):
        bot = 0
        top = 0
        for n in range(ln):
            if l == n:
                continue
            bot += math.exp(-(a[0][0]*((X[n][0]-X[l][0])**2))-(2*a[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(a[1][1]*((X[n][1]-X[l][1])**2)))
            top += math.exp(-(a[0][0]*((X[n][0]-X[l][0])**2))-(2*a[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(a[1][1]*((X[n][1]-X[l][1])**2)))*y[n]
        total += (y[l] - (top/bot))**2
    return total

def knn_loss(a, k):
    ototal = 0
    yn = y_df.copy()
    
    for l in range(ln):
        knn = []
        total = 0
        for n in range(ln):
            if l == n:
                continue
            temp =  math.exp(-(a[0][0]*((X[n][0]-X[l][0])**2))-(2*a[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(a[1][1]*((X[n][1]-X[l][1])**2)))
            knn.append([temp, y[n]])
        knn.sort(key = sF)
        for i in range(min(k, ln - 1)):
            total += knn[i][1]
        total = total/k
        ototal += (y[l] - total)**2
    return ototal

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    w1 = np.array([[1,0],
              [0,1]])*alpha
    yn = maha2(w1)
    print("Loss  =  " + str(kernel_loss(w1)))
    return yn

def sF(val): 
    return val[0]

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    alpha = 1
    w1 = np.array([[1,0],
              [0,1]])*alpha
    yn = y_df.copy()
    
    for l in range(ln):
        knn = []
        total = 0
        for n in range(ln):
            if l == n:
                continue
            temp =  math.exp(-(w1[0][0]*((X[n][0]-X[l][0])**2))-(2*w1[0][1]*(X[n][0]-X[l][0])*(X[n][1]-X[l][1]))-(w1[1][1]*((X[n][1]-X[l][1])**2)))
            knn.append([temp, y[n]])
        knn.sort(key = sF)
        for i in range(min(ln - 1, k)):
            total += knn[i][1]
        yn[l] = (total/k)
    print("Loss  =  " + str(knn_loss(w1, k)))
    return yn

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    plot_kernel_preds(alpha)

for k in (1, 5, 15):
    plot_knn_preds(k)
