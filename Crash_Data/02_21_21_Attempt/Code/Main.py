import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Feature_Engineering_3 import Feature_Engineering
from Get_Labels import Get_Labels


def Main():
    X, Y, Crash, Column_Contents = Feature_Engineering()
    Labels = Get_Labels(Column_Contents)
    print ("Fit")
    reg = SGDRegressor()
#    reg = SGDClassifier()
    reg.fit(X,Y)
    C = reg.coef_
#    C = C[0] # Use this if using SGDClassifier
    print (C)
    print (len(C))
    I = reg.intercept_
    s = -1
#    for a in Column_Contents:
#        print ()
#        for b in a[1]:
#            s += 1
#            if abs(C[s]) > 0.0:
#                print ("%s %s %s %5.4f" % (a[0], b, Labels[s][2], C[s]))

    print ()
    n = -1
    for a in Column_Contents:
        print ()
        A = []
        for b in a[1]:
            n += 1
            B = [a[0], b, Labels[n][2], C[n]]
            A.append(B)
        A = sorted(A, key=lambda x:x[3], reverse=True)
        for row in A:
            print ("%5.4f & \\verb|%s| & %s & %s \\cr" % (row[3], row[0], row[1], row[2]))
        print ("%5.4f & \\verb|%s| & %s & %s \\cr" % (sum([x[3] for x in A]), row[0], "Sum", ""))

    print ()
    A = []
    n = -1
    for a in Column_Contents:
        for b in a[1]:
            n += 1
            A.append([a[0], b, Labels[n][2], C[n]])
    A = sorted(A, key=lambda x:x[3], reverse=True)
    for row in A[:20]:
        print ("%5.4f %s %s %s" % (row[3], row[0], row[1], row[2]))
    print ()
    for row in A[-20:]:
        print ("%5.4f %s %s %s" % (row[3], row[0], row[1], row[2]))

        
        
Main()
