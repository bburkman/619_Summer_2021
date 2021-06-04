import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import math

def SavePlot(X, Y, w_t, w, count):
    filename = './Plots/Plot_' + str(count) + '.png'
    Red = X[Y[:]==-1, :]
#    print (Red)
#    print ()
    Blue = X[Y[:]==1, :]
#    print (Blue)
    plt.axis((0,1,0,1))
    plt.scatter(Red[:,1], Red[:,2], c='r')
    plt.scatter(Blue[:,1], Blue[:,2], c='b')
    plt.plot([0,1],[(w_t[0] + 0*w_t[1])/(-w_t[2]), (w_t[0] + 1*w_t[1])/(-w_t[2])], c='g')
    plt.plot([0,1],[(w[0] + 0*w[1])/(-w[2]), (w[0] + 1*w[1])/(-w[2])], c='r')

    plt.savefig(filename)
    plt.close()

def Generate_Linearly_Separable_Data(w, n):

    X = np.empty((0,3))
    Y = np.empty(0)
    for i in range (n):
        x = np.array([1,random.random(),random.random()])
        X = np.vstack((X,x))
        if np.dot(w,x)>0:
            y = np.array(-1)
        else:
            y = np.array(1)
        Y = np.append(Y,y)

    for i, x in enumerate(X):
        print (x, Y[i])
    return X, Y
    
def Perceptron(X, Y, w, gamma):
    A = np.matmul(X,w)
#    print ()
#    print (A)
    A = np.multiply(A,Y)
#    print ()
#    print (A)
#    print ()
    B = [i for i, a in enumerate(A) if A[i]<0]
#    print (B)
    if len(B)>0:
        i = B[0]
        w = w  + gamma*Y[i]*X[i]
#    if w[0] !=0:
#        w = w/(-w[0])
#    print (len(B), w/w[0])

    return w, len(B)
    
            

def Main():
    # Target function
    w_t = np.array([1,-1,-1])
    n = 100
    gamma = 0.01
    X, Y = Generate_Linearly_Separable_Data(w_t,n)
    # Initial Guess
    w = np.array([-0.9,0,1])
    i=0
    b = 1
    while b>0:
        w, b = Perceptron(X, Y, w, gamma)
        if i%10==0:
            SavePlot(X, Y, w_t, w, i)
        print (i, b, w/w[0])
        i += 1
    SavePlot(X, Y, w_t, w, i)


Main()


