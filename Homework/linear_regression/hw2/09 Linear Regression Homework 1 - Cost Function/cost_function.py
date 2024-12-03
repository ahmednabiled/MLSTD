import numpy as np

def cost(t,X,W) -> float:
    pred = np.dot(X,W)
    error = pred - t
    cost = error.T @ error / (2*X.shape[0])
    return cost


def fdriv(t,X,W): 
    X = np.array(X)
    pred = np.dot(X,W)
    error = pred - t
    gradient = X.T @ error / X.shape[0]
    return gradient



if __name__ == "__main__":
    # X = np.array([[1,2,3],[4,5,6]])
    # W = np.array([1,2,3])
    # print(np.dot(X,W))
    # #x[2][3] W[3][1]

    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = 5 + X   

    X = X.reshape((-1, 1))  
    X = np.hstack([np.ones((X.shape[0], 1)), X])  

    print(X.shape)  
    weights = np.array([1.0, 1.0])  
    print(cost(t, X, weights))
    print(fdriv(t, X, weights)) 

    #res
    """
    (5, 2)
    8.0
    [-4.   -1.92]
    """