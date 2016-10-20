from numpy import array, dot, log, e, ndarray, append, newaxis
from random import random
from copy import copy
from itertools import chain

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
        if len(layernum) < 3:
            raise ValueError('param layernum is too small.'
                             'It should consist input/hidden/output layer')
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

    def fowardprop(self, inp, return_a=False):
        if isinstance(inp, list):
            inp = array(inp)
        if len(inp) != self.layernum[0]:
            raise RuntimeError('input size doesn\'t match')
        if not isinstance(inp, ndarray):
            raise TypeError('input is not an ndarray')
        if return_a:
            a = []
        for l in range(self.layercount - 1):
            inp = append(array([1]), inp)  # Add bias unit
            start, end = self.partialthetalen[l], self.partialthetalen[l+1]
            thetaseg = self.theta[start: end]
            inp = _fowardprop(thetaseg, inp)
            if return_a:
                a.append(inp)
        if return_a:
            return inp, a
        else:
            return inp

    def costfunc(self, out, ans):
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
        self.backprop(out, ans, a)
