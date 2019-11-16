import numpy as np

def activation_fn(x):
    f = 1/(1+np.exp(-x)) #-sigmoid fn
    return f

def weighted_biased_act_fn(x,w=0,b=0):
    f= 1/(1+np.exp(-(x*w+b)))
    return f


