#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
  
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
  
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs)*y) / N
    dx = probs.copy()
    dx -= y
    dx /= N
    return loss, dx

def gradcheck(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx = f(x) # Evaluate function value at original point
    h = 1e-4

    eps = 1e-5


    numgrad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore
        numgrad[ix] = (fxph - fxmh) / (2 * h) # the slope
        it.iternext() # step to next dimension

    return numgrad


def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    labels = labels.astype("int64")
    a_1_0 = data.dot(W1) + b1
    a_1   = sigmoid(a_1_0)
    a_2_0 = (a_1.dot(W2) + b2)

    loss, dx = softmax_loss(a_2_0, labels)

    gradb2 = np.sum(dx, axis=0, keepdims=True)
    gradW2 = a_1.T.dot(dx)
    da_1 = sigmoid_grad(a_1)*dx.dot(W2.T)
    gradb1 = np.sum(da_1, axis=0, keepdims=True)
    gradW1 = data.T.dot(da_1)

    # fb2 = lambda x: (softmax_loss(a_1.dot(W2)+x, labels)[0])
    # print "+++++++++++++++++++++++++++"
    # print gradb2
    # print "---------------------------"
    # print gradcheck(fb2,b2)
    # print "***************************"

    # fW2 = lambda x: (softmax_loss(a_1.dot(x)+b2, labels)[0])
    # print "+++++++++++++++++++++++++++"
    # print gradW2
    # print "---------------------------"
    # print gradcheck(fW2,W2)
    # print "***************************"

    assert(gradb2.shape == b2.shape)
    assert(gradW2.shape == W2.shape)
    assert(gradb1.shape == b1.shape)
    assert(gradW1.shape == W1.shape)

    cost = loss
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()