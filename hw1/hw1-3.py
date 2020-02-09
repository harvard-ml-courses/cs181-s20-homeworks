import math
from math import exp
import numpy as np
    
alpha = 10

x1 = np.array([0,0,0,.5,.5,.5,1,1,1])
x2 = np.array([0,.5,1,0,.5,1,0,.5,1])
y = np.array([0,0,0,.5,.5,.5,1,1,1])

ln = len(x1)

w1 = np.array([[1,0],
              [0,1]])*alpha

w2 = np.array([[0.1,0],
              [0,1]])*alpha

w3 = np.array([[1,0],
              [0,0.1]])*alpha

def maha2(d):
    total = 0
    for l in range(ln):
        bot = 0
        top = 0
        for n in range(ln):
            if l == n:
                continue
            bot += math.exp(-(d[0][0]*((x1[n]-x1[l])**2))-(2*d[0][1]*(x1[n]-x1[l])*(x2[n]-x2[l]))-(d[1][1]*((x2[n]-x2[l])**2)))
            top += math.exp(-(d[0][0]*((x1[n]-x1[l])**2))-(2*d[0][1]*(x1[n]-x1[l])*(x2[n]-x2[l]))-(d[1][1]*((x2[n]-x2[l])**2)))*y[n]
        total += (y[l] - (top/bot))**2

    print(total)

print("Loss of W1 =")
maha2(w1)
print("Loss of W2 =")
maha2(w2)
print("Loss of W3 =")
maha2(w3)
