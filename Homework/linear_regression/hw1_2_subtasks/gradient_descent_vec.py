import numpy as np
from numpy.linalg import norm

def gradient_decent(fderiv,initial_point,max_iter=1000,precision=0.001):
    x_cur = np.array(initial_point)
    x_old = x_cur + precision*200
    iter=0
    while norm(x_cur-x_old) > precision and iter < max_iter:
        x_old = x_cur.copy()
        gradient = fderiv(x_cur)
        x_cur = x_cur - gradient * precision
        iter += 1

    return x_cur

