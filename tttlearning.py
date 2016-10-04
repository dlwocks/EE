from numpy import array, dot, log, e
from scipy.optimize import minimize
from random import random

from ttttester import complete_check
from logreg_ai import logreg_ai


def costfunc_aep(theta):
    return (2 - complete_check(logreg_ai(theta).getstep))


testdata = array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0]])
testans = array([1, 1, 1, 1, 1, 0])

if __name__ == '__main__':
    # dk = complete_check()
    # data, ans = dk.fetch()
    theta = array([0 for i in range(9)])
    print(theta)
    # theta = minimize(costfunc, theta, args=(data, ans), jac=costfunc_d, method='BFGS')
    theta = minimize(costfunc_aep, theta).x
    print(theta)
    ai = logreg_ai(theta)
    complete_check(ai.getstep, pt=True)

