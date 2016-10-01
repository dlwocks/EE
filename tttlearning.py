from numpy import array, dot, log, e
from scipy.optimize import minimize
from random import random

from ttttester import exhaustive_check


def sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    return sum(ans * -log(sigmoid(dot(data, theta))) - (1 - ans) * log(1 - sigmoid(dot(data, theta))))


def costfunc_d(theta, data, ans):
    return dot(data.T, (sigmoid(dot(data, theta)) - ans)) / len(theta)


testdata = array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0]])
testans = array([1, 1, 1, 1, 1, 0])

if __name__ == '__main__':
    dk = exhaustive_check()
    data, ans = dk.fetch()
    theta = array([0 for i in range(9)])
    print(theta)
    theta = minimize(costfunc, theta, args=(data, ans), jac=costfunc_d, method='BFGS')
    print(theta)
