from numpy import array, dot, log, e, ndarray, append, newaxis
from random import random
from copy import copy
from itertools import chain
from functools import reduce
from scipy.optimize import minimize

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n; that it is for first node in FORMER layer
'''
def debug(message, param):
    print(message +': ' + str(param))


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
            # self.theta = array([(random()-0.5)/100 for i in range(self.totalthetalen)])
            self.theta = array([0 for i in range(self.totalthetalen)])

    def fowardprop(self, allinp, return_a=False, return_out=False):
        if isinstance(allinp, list):
            allinp = array(allinp)
        elif not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray or a list')
        if return_a:
            a = []
        if return_out:
            out = []
        for inp in allinp:
            temp_a = [inp]
            # debug('inputsize', inp)
            if len(inp) != self.layernum[0]:
                raise RuntimeError('input size doesn\'t match')
            for l in range(self.layercount - 1):
                inp = append(array([1]), inp)  # Add bias unit
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                thetaseg = self.theta[start: end]
                inp = _fowardprop(thetaseg, inp)
                if return_a:
                    if l == self.layercount - 2:
                        temp_a.append(inp)
                    else:
                        temp_a.append(append(array([1]), inp))
            if return_a:
                a.append(temp_a)
            if return_out:
                out.append(inp)
        if return_a and return_out:
            return out, a
        elif return_out:
            return out
        elif return_a:
            return a
        else:
            raise RuntimeError('fowardprop call without expecting any return')

    def costfunc(self, _, inp, ans):
        out = array(self.fowardprop(inp, return_out=True))
        return sum(sum(ans * -log(out).T - (1 - ans) * log(1 - out).T))

    def gradient_single(self, inp, ans):
        inp = array([inp])
        a = self.fowardprop(inp, return_a=True)[0]
        lasterror = a[-1] - ans
        delta = list(chain.from_iterable(a[-2][None].T * lasterror[None]))  # None == numpy.newaxis
        for i in range(self.layercount - 2, 0, -1):  # It is backprop!
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = self.theta[start: end].reshape(self.layernum[i]+1, self.layernum[i+1])
            thiserror = dot(thetaseg, lasterror) * (a[i] * (1 - a[i]))
            lasterror = thiserror
            delta = list(chain.from_iterable(a[i-1][None].T * lasterror[None])) + delta
        return array(delta)

    def gradient(self, _, inp, ans):
        return [i / len(ans) for i in reduce(lambda a, b: a + b, [self.gradient_single(thisinp, thisans) for thisinp, thisans in zip(inp, ans)])]

    def train(self, inp, ans):
        self.theta = minimize(self.costfunc, self.theta, args=(inp, ans), jac=self.gradient, method='BFGS').x
