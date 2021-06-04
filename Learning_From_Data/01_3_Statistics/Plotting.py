import matplotlib.pyplot as plt
import math
import numpy as np

# Binomial Distribution
# n = number of trials
# p = success probability for each trial
# q = 1-p
# k = number of successes

def Plot():
    n = 10
    p = 0.9
    q = 1-p
    
    X = [k for k in range (1,n+1)]
    Y = [math.comb(n,k) * p**k * q**(n-k) for k in X]
    Z = [2 * math.exp(-2 * n * (p-k/10)**2) for k in X] 
    #    print (X)
    print (sum(Y))
    plt.scatter(X,Y)
    plt.scatter(X,Z)
    plt.show()



Plot()
