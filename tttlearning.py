from numpy import array, dot, log, e
from scipy.optimize import fmin


def _sigmoid(z):
    return 1/(1+e**(-z))


def sigmoid(theta, data):
    return _sigmoid(dot(data, theta))


def costfunc(theta, data, ans):
    sum(-ans * log(sigmoid(theta, data)) - (1 - ans) * log(1-sigmoid(theta, data)))

