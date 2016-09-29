from numpy import array, dot, log, e
from scipy.optimize import fmin


def _sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    return sum(ans * -log(_sigmoid(dot(data, theta))) - (1 - ans) * log(1 - _sigmoid(dot(data, theta))))

