import numpy as np 

def fderiv(x):
    return 2*(x-2)


def gradient_decent(fderiv,initial_point=0,step_size=0.001,no_iter = 100000):
    
    x = initial_point
    gradient = fderiv(x)

    for i in range(no_iter):
        gradient = fderiv(x)
        x = x - gradient * step_size
    
    return x
    
print(gradient_decent(fderiv,0))