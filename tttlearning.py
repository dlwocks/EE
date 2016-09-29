from numpy import array, dot, log, e
from scipy.optimize import fmin
from random import random

from ttttester import exhaustive_check


def _sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    return sum(ans * -log(_sigmoid(dot(data, theta))) - (1 - ans) * log(1 - _sigmoid(dot(data, theta))))


if __name__ == '__main__':
    dk = exhaustive_check()
    data, ans = dk.fetch_data_array()
    theta = array([(random()-0.5)/1000 for i in range(9)])
    print(theta)
    theta = fmin(costfunc, theta, args=(data, ans),maxfun=100000)
    print(theta)
