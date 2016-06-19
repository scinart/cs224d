#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

# First implement a gradient checker by filling in the following functions

def save_random_state():
    return (random.getstate(), np.random.get_state())
def load_random_state(rnd_tuple):
    rndstate, np_rndstate = rnd_tuple
    random.setstate(rndstate)
    np.random.set_state(np_rndstate)

def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rnd_tuple = save_random_state()

    fx, grad = f(x) # Evaluate function value at original point
    load_random_state(rnd_tuple)

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
        fxph = f(x)[0] # evalute f(x + h)

        # f may change random state.
        # we have to set it back.
        load_random_state(rnd_tuple)

        x[ix] = oldval - h
        fxmh = f(x)[0] # evaluate f(x - h)
        load_random_state(rnd_tuple)
        x[ix] = oldval # restore
        numgrad = (fxph - fxmh) / (2 * h) # the slope

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > eps or np.isinf(reldiff) or np.isnan(reldiff):
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    x = np.random.randn(3,)
    gradcheck_naive(quad, x)    # 1-D test
    x = np.random.randn(4,5)
    gradcheck_naive(quad, x)   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
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
