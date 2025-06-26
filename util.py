import numpy as np

def mse(Y_true, Y_pred):
    Y_true, Y_pred = (Y_true.flatten(), Y_pred.flatten())
    m = len(Y_true)
    return (1/m)*np.sum( (Y_true - Y_pred) ** 2)

def rss(Y_true, Y_pred):
    Y_true, Y_pred = (Y_true.flatten(), Y_pred.flatten())
    return np.sum( (Y_true - Y_pred) ** 2 )

def tss(Y_true):
    Y_true = Y_true.flatten()
    return np.sum( (Y_true - np.mean(Y_true)) ** 2)

def r2(Y_true, Y_pred):
    Y_true, Y_pred = (Y_true.flatten(), Y_pred.flatten())
    return 1 - (rss(Y_true, Y_pred) / tss(Y_true))