from gradient_descent_vec import gradient_decent
from math import *
import numpy as np


def trail():
    def f(vec):
        return sin(vec[0])+cos(vec[1])+sin(vec[2])

    def partial_x(x):
        return cos(x)

    def partial_y(y):
        return -sin(y)

    def partial_z(z):
        return cos(z)

    def fderiv(vec):
        return np.array([partial_x(vec[0]),partial_y(vec[1]),partial_z(vec[2])])
    
    res = gradient_decent(fderiv,(1,2,3.5))
    return (res ,f(res))

if __name__=="__main__":
    print(trail()[0])
    print(trail()[1])


"""
results :
[0.22457445 2.67782963 4.21311734]
-1.5496155799445872
"""
