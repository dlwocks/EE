'''
A general implementation of ANN cython version of the one defined in learningfunc.py

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
import numpy as np
cimport numpy as np
from random import uniform
from copy import copy
from itertools import accumulate
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
    inittheta = np.array([])
    for outnum in layernum:
        if inpnum:
            eps_init = np.sqrt(6)/np.sqrt(inpnum+outnum)
            appendedtheta = np.array([uniform(-eps_init, eps_init)
                                   for i in range((inpnum+1)*outnum)])
            inittheta = np.append(inittheta, appendedtheta)
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
        self.unitnum = sum(layernum)
        self.layercount = len(layernum)
        self.cumlayernum = [0] + list(accumulate(layernum))[:-1]
        if theta is not None:
            if not isinstance(theta, np.ndarray):
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
        if self.fpcache_enabled and self.fpcache is not None and np.array_equal(self.fpcache_theta, theta):
            return self.fpcache
        if theta is None:
            theta = self.theta
        if not isinstance(allinp, np.ndarray):
            raise TypeError('input is not a ndarray')
        a = np.empty((len(allinp), self.unitnum))
        for l in range(self.layercount - 1):
            a[:,self.cumlayernum[l]:self.cumlayernum[l+1]] = allinp
            start, end = self.partialthetalen[l], self.partialthetalen[l+1]
            bias = theta[start:start+self.layernum[l+1]]
            thetaseg = theta[start+self.layernum[l+1]: end].reshape(self.layernum[l], self.layernum[l+1])
            allinp = sigmoid(np.dot(allinp, thetaseg) + bias)
        a[:,self.cumlayernum[-1]:] = allinp
        if self.fpcache_enabled:
            self.fpcache_theta = theta.copy()
            self.fpcache = a
        return a

    def get(self, inp):
        return self.fowardprop(np.array([inp]))[0, self.cumlayernum[-1]:]  # [0]: first inp's output(while there's only one for .get)

    def costfunc(self, theta, inp, ans):
        out = self.fowardprop(inp, theta)[:, self.cumlayernum[-1]:]
        return (ans * -np.log(out) - (1 - ans) * np.log(1 - out)).sum()

    def gradient(self, theta, inp, ans):
        a = self.fowardprop(inp, theta)
        g = np.zeros_like(theta)
        ln = self.layernum
        cln = self.cumlayernum
        fillptr = self.totalthetalen
        lasterror = a[:, self.cumlayernum[-1]:] - ans
        for i in range(len(a)):
            g[fillptr - ln[-2] * ln[-1]: fillptr] += (a[i][cln[-2]:cln[-1]][None].T * lasterror[i][None]).flatten()
        fillptr -= ln[-2] * ln[-1]
        g[fillptr - ln[-1]: fillptr] = np.sum(lasterror, axis=0)
        fillptr -= ln[-1]
        for l in range(self.layercount - 2, 0, -1):
            start, end = self.partialthetalen[l], self.partialthetalen[l+1]
            thetaseg = theta[start+ln[l+1]: end].reshape(ln[l], ln[l+1])
            d = np.inner(lasterror, thetaseg)
            aseg = a[:,self.cumlayernum[l]: self.cumlayernum[l+1]]
            agrad = aseg * (1 - aseg)
            lasterror = d * agrad
            for i in range(len(a)):
                g[fillptr - ln[l-1] * ln[l]: fillptr] += (a[i][cln[l-1]: cln[l]][None].T * lasterror[i][None]).flatten()
            fillptr -= ln[l-1] * ln[l]
            g[fillptr - ln[l]: fillptr] = np.sum(lasterror, axis=0)
            fillptr -= ln[l]
        assert fillptr == 0, 'fillptr should be 0, but is %d' % fillptr
        g /= len(ans)
        return g

    def train(self, inp, ans, gtol=1e-5):
        if not (isinstance(inp, np.ndarray) and isinstance(ans, np.ndarray)):
            raise TypeError(str(type(inp)), str(type(ans)))
        if not len(inp.shape) == 2:
            raise ValueError('input is not 2d')
        if not len(ans.shape) == 2:
            raise ValueError('answer is not 2d')
        if not len(inp) == len(ans):
            raise ValueError('different number of input and answer')
        if not len(inp[0]) == self.layernum[0]:
            raise ValueError('single input\'s size doesn\'t match with input unit number')
        if not len(ans[0]) == self.layernum[-1]:
            raise ValueError('single output\'s size doesn\'t match with output unit number')
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
        self.fpcache = None
        self.theta = minres.x
        return minres
