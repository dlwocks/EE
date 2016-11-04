'''
A Cython version of ANN defined in learningfunc.py
'''
cimport numpy as np
from numpy import array, dot, log, ndarray, append, sqrt, array_equal, zeros_like
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
        if self.fpcache_enabled and self.fpcache and array_equal(self.fpcache_theta, theta):
            return self.fpcache
        if theta is None:
            theta = self.theta
        if not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray')
        a = []
        for inp in allinp:
            temp_a = []
            if len(inp) != self.layernum[0]:
                raise RuntimeError('input size doesn\'t match. length of input is %d, while it should be %d' % (len(inp), self.layernum[0]))
            for l in range(self.layercount - 1):
                temp_a.append(inp)  # Append bias UNADDED layer to temp_a
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                bias = theta[start:start+self.layernum[l+1]]
                thetaseg = theta[start+self.layernum[l+1]: end]
                inp = sigmoid(dot(inp, thetaseg.reshape(self.layernum[l], self.layernum[l+1])) + bias)
            temp_a.append(inp)
            a.append(temp_a)
        if self.fpcache_enabled:
            self.fpcache_theta = copy(theta)
            self.fpcache = a
        return a

    def get(self, inp):
        return self.fowardprop(array([inp]))[0][-1]  # [0]: first inp's output(while there's only one for .get)

    def costfunc(self, theta, inp, ans):
        out = array([d[-1] for d in self.fowardprop(inp, theta)])
        totalcost = 0
        for tout, tans in zip(out, ans):
            totalcost += sum(tans * -log(tout) - (1 - tans) * log(1 - tout))
        return totalcost

    def gradient_single(self, theta, a, ans):
        lasterror = a[-1] - ans
        delta = list(lasterror) + list((a[-2][None].T * lasterror[None]).flatten())
        for i in range(self.layercount - 2, 0, -1):
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = theta[start+self.layernum[i+1]: end].reshape(
                self.layernum[i], self.layernum[i+1])
            d = dot(thetaseg, lasterror)
            agrad = a[i] * (1 - a[i])
            lasterror = d * agrad  # This is in fact this(ith) layer's error; below same.
            subdelta = list(lasterror) + list((a[i-1][None].T * lasterror[None]).flatten())
            delta = subdelta + delta
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