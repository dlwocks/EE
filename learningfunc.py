from numpy import array, dot, log, e, ndarray, append, newaxis
from random import random
from copy import copy
from itertools import chain
from scipy.optimize import minimize

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n; that it is for first node in FORMER layer
'''

def sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    sig = sigmoid(dot(data, theta))
    return sum(ans * -log(sig) - (1 - ans) * log(1 - sig))


def costfunc_d(theta, data, ans):
    return dot(data.T, (sigmoid(dot(data, theta)) - ans)) / len(theta)


def gen_piece(board):
    '''
    input board: 1d, list
    '''
    temp = [0 for i in range(9)]
    ret = []
    for i in range(1, 10):
        try:
            temp[board.index(i)] = i
        except ValueError:
            if i <= 5:
                raise  # The game couldn't have ended!
            break
        ret.append(copy(temp))
    return ret


def _thetalen(layernum):
    total = 0
    partiallen = [0]
    temp = None
    for l in layernum:
        if l <= 0 or not isinstance(l, int):
            raise RuntimeError('a layernum is not an positive integer')
        if temp:
            total += (temp + 1) * l
            partiallen.append(total)
        temp = l
    return partiallen


def _fowardprop(theta, data):
    '''
    theta: partial theta in the inbetween region, in array
    data: input (should be in array)
    '''
    outputsize = len(theta) // len(data)
    z = dot(data, theta.reshape(len(data), outputsize))
    a = sigmoid(z)
    return a


class ann(object):
    def __init__(self, layernum, theta=None):
        if not isinstance(layernum, list):
            raise TypeError('param layernum is not list')
        if len(layernum) < 2:
            raise ValueError('param layernum is too small.'
                             'It should at least consist input/output layer')
        self.partialthetalen = _thetalen(layernum)
        self.totalthetalen = self.partialthetalen[-1]
        self.layernum = layernum
        self.layercount = len(layernum)
        if theta:
            if not isinstance(theta, ndarray):
                raise TypeError('param theta, though inputted, is not an array')
            if len(theta) != len(self.layercount):
                raise ValueError('length of theta is not what it should be accd. to layernum')
            self.theta = theta
        else:
            self.theta = array([(random()-0.5)/100 for i in range(self.totalthetalen)])

    def fowardprop(self, allinp, return_a=False, return_res=True):
        if isinstance(allinp, list):
            allinp = array(allinp)
        elif not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray or a list')
        if return_a:
            a = []
        if return_res:
            res = []
        for inp in allinp:
            temp_a = []
            if len(inp) != self.layernum[0]:
                raise RuntimeError('input size doesn\'t match')
            for l in range(self.layercount - 1):
                inp = append(array([1]), inp)  # Add bias unit
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                thetaseg = self.theta[start: end]
                inp = _fowardprop(thetaseg, inp)
                if return_a:
                    temp_a.append(inp)
            if return_a:
                a.append(temp_a)
            if return_res:
                res.append(inp)
        if return_a and return_res:
            return inp, a
        elif return_res:
            return inp
        elif return_a:
            return a

    def costfunc(self, out, ans, _):
        return sum(ans * -log(out) - (1 - ans) * log(1 - out))

    def gradient_single(self, out, ans, a):
        lasterror = a[-1] - ans
        delta = list(chain.from_iterable(a[self.layercount-1][None].T * lasterror[None])) # None == numpy.newaxis
        for i in range(self.layercount - 1, 1, -1):  # It is backprop!
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = self.theta[start: end]
            thiserror = (dot(thetaseg.reshape(self.layernum[i]+1, self.layernum[i+1]), lasterror)) * (a[i] * (1 - a[i]))
            lasterror = thiserror
            delta = list(chain.from_iterable(a[i-1][None].T * lasterror[None])) + delta
        return delta

    def gradient(self, out, ans, a):
        return sum([self.gradient_single(thisout, thisans, a) for thisout, thisans in zip(out, ans)])

    def train(self, inp, ans):
        out, a = self.fowardprop(inp, return_a=True)
        self.theta = minimize(self.costfunc, self.theta, args=(out, ans, a), jac=self.gradient, method='BFGS').x
