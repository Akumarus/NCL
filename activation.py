import numpy as np


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """
    Derivate of the sigmoid function.

    :param y:
    :return:
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    """
    Hyperbolic tangent

    :param x:
    :return:
    """
    return np.tanh(x)


def tanh_der(x):
    """
    Derivate of the hyperbolic tangent function.

    :param x:
    :return:
    """
    return 1.0 - np.power(tanh(x), 2)


def leaky_relu(x):
    """
    Leaky rectified linear unit.

    :param x:
    :return:
    """
    return np.minimum(0.01 * x, x)


def leaky_relu_der(x):
    """
    Derivate of leaky relu.

    :param x:
    :return:
    """
    y = np.ones_like(x)
    y[x > 0] = 0.01
    return y

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    why the max - see Eli's post https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    e_x = np.exp(x - np.max(x)) # 
    return e_x / e_x.sum()
  
def derv_softmax(s): # where x is the input
    """
    x is input vector of shape 1* N
    and x is the softmax vector of x with same shape as x
    Derivative of softmax is a Jacobian matrix of size N^2
    we have to take derivative of softmax for each input x = 1 to N
    And since softmax is a vector we need to take derivative of each element in vector  1 tto N
    
    Assuming s.shape == x.shape (3) then the Jacobian (Jik) of the derivative is given below (shape is np.diag(s))
    ds1/dx1 ds1/dx2 ds1/dx3
    ds2/dx1 ds2/dx2 ds2/dx3
    ds3/dx1 ds3/dx2 ds3/dx3
    Note - we dont know x vector; but that's fine; as dsk/dxi= sk(kronecker_ik - si) from (2)
    whe sk is softmax of kth element and si of ith element ( and remember we are working with softmax input)
    Here k goes from 1 to N in outer loop
      and     i goes from 1 to N in inner loop to give us the above Jacobian ik
    (1) https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    (2) https://e2eml.school/softmax.html
    (3) https://stackoverflow.com/a/46028029/429476
    """
    N = len(s)
    Jik =  np.zeros(shape=(N,N))
    for k in range(0, N):
        for i in range(0, N):
            kronecker_ik = 1 if i ==k else 0
            Jik[k][i] = s[k]* (kronecker_ik -s[i]) 
    return Jik

fun_dict = {'sigmoid': {'activation': sigmoid,
                        'derivative': sigmoid_der},
            'tanh': {'activation': tanh,
                     'derivative': tanh_der},
            'linear': {'activation': lambda x: x,
                       'derivative': lambda x: 1.0},
            'leaky_relu': {'activation': leaky_relu,
                           'derivative': leaky_relu_der},
            'softmax': {'activation': softmax,
                           'derivative': derv_softmax}}