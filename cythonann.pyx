'''
A general implementation of ANN; cython version of the one defined in learningfunc.py

How to use this in a script:
# Import as following
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from cythonann import ann
# initialize as following
ann = cythonann.ann([1, 2, 3, 4]) # Where there is 1 input unit, 2 hidden layer with 2, 3 hidden unit respectively, 4 output unit
ann.train(inp, ans) # Where inp, ans are 2d-ndarray, with dimension (# of training case) * (# of inp(inp)/out(ans) unit)
ann.get(inp) # Where inp is 1d-ndarray of one test data
'''
cimport numpy as np
from numpy import array, dot, log, ndarray, append, sqrt, array_equal, zeros_like, empty_like, empty
from random import uniform
from copy import copy
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
import warnings


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



def _rndinit(layernum):
    inpnum = None
    inittheta = array([])
    for outnum in layernum:
        if inpnum:
            eps_init = sqrt(6)/sqrt(inpnum+outnum)
            appendedtheta = array([uniform(-eps_init, eps_init)
                                   for i in range((inpnum+1)*outnum)])
            inittheta = append(inittheta, appendedtheta)
        inpnum = outnum
    return inittheta


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
        if theta is not None:
            if not isinstance(theta, ndarray):
                raise TypeError(
                    'param theta, though inputted, is not an array')
            if len(theta) != self.partialthetalen[-1]:
                raise ValueError(
                    'length of theta should be %d, but inputted %d' %
                    (self.partialthetalen[-1], len(theta)))
            self.theta = theta
        else:
            self.theta = _rndinit(layernum)
        self.fpcache_enabled = False
        self.fpcache_theta = None
        self.fpcache = None

    def fowardprop(self, allinp, theta=None):
        if self.fpcache_enabled and self.fpcache is not None and array_equal(self.fpcache_theta, theta):
            return self.fpcache
        if theta is None:
            theta = self.theta
        if not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray')
        a = empty(len(allinp), dtype='object')
        for i, inp in enumerate(allinp):
            single_a = []
            if len(inp) != self.layernum[0]:
                raise RuntimeError('input size doesn\'t match. length of input is %d, while it should be %d' % (len(inp), self.layernum[0]))
            for l in range(self.layercount - 1):
                single_a.append(inp)  # Add bias UNADDED layer to single_a
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                bias = theta[start:start+self.layernum[l+1]]
                thetaseg = theta[start+self.layernum[l+1]: end].reshape(self.layernum[l], self.layernum[l+1])
                inp = sigmoid(dot(inp, thetaseg) + bias)  # equiv. to bias * 1: add it to add bias to input
            single_a.append(inp)
            a[i] = single_a
        assert i == len(allinp) - 1
        if self.fpcache_enabled:
            self.fpcache_theta = theta.copy()
            self.fpcache = a
        return a

    def get(self, inp):
        return self.fowardprop(array([inp]))[0][-1]  # [0]: first inp's output(while there's only one for .get)

    def costfunc(self, theta, inp, ans):
        out = array([d[-1] for d in self.fowardprop(inp, theta)])
        return (ans * -log(out) - (1 - ans) * log(1 - out)).sum()

    def gradient_single(self, theta, a, ans):
        fillptr = self.totalthetalen
        ln = self.layernum
        delta = empty_like(theta)
        lasterror = a[-1] - ans
        delta[fillptr - ln[-2] * ln[-1]: fillptr] = (a[-2][None].T * lasterror[None]).flatten()
        fillptr -= ln[-2] * ln[-1]
        delta[fillptr - ln[-1]: fillptr] = lasterror
        fillptr -= ln[-1]
        for i in range(self.layercount - 2, 0, -1):
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = theta[start+self.layernum[i+1]: end].reshape(
                self.layernum[i], self.layernum[i+1])
            d = dot(thetaseg, lasterror)
            agrad = a[i] * (1 - a[i])
            lasterror = d * agrad  # This is in fact this(ith) layer's error; below same.
            delta[fillptr - ln[i-1] * ln[i]: fillptr] = (a[i-1][None].T * lasterror[None]).flatten()
            fillptr -= ln[i-1] * ln[i]
            delta[fillptr - ln[i]: fillptr] = lasterror
            fillptr -= ln[i]
        assert fillptr == 0, 'fillptr should be 0, but is %d' % fillptr
        return array(delta)

    def gradient(self, theta, inp, ans):
        PARALLEL = True   # Parallel learning shows better convergence.
        if PARALLEL:
            a = self.fowardprop(inp, theta)
            g = zeros_like(theta)
            for thisa, thisans in zip(a, ans):
                g += self.gradient_single(theta, thisa, thisans)
            g /= len(ans)
        else:  # Series Delta
            init_theta = copy(theta)
            for thisinp, thisans in zip(inp, ans):
                grad = self.gradient_single(theta, thisinp, thisans) / len(ans)
                theta = theta + grad
            g = theta - init_theta
        return g

    def train(self, inp, ans, gtol=1e-5):
        if not (isinstance(inp, ndarray) and isinstance(ans, ndarray)):
            raise TypeError(str(type(inp)), str(type(ans)))
        if not (len(inp.shape) == 2 and len(ans.shape) == 2 and len(inp) == len(ans) and len(inp[0]) == self.layernum[0] and len(ans[0]) == self.layernum[-1]):
            raise ValueError((len(inp.shape) == 2, len(ans.shape) == 2, len(inp) == len(ans), len(inp[0]) == self.layernum[0], len(ans[0]) == self.layernum[-1]))
        self.fpcache_enabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minres = minimize(self.costfunc,
                              self.theta,
                              args=(inp, ans),
                              jac=self.gradient,
                              method='BFGS',
                              options={'gtol': gtol})
        self.fpcache_enabled = False
        self.theta = minres.x
        return minres
