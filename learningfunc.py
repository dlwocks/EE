'''
Miscellaneous functions and classes related to machine learning.
'''
from numpy import array, dot, log, ndarray, append, sqrt, array_equal, zeros_like, isclose, split, squeeze, empty, empty_like
import numpy as np
from random import uniform
from copy import copy
from itertools import count, accumulate
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
import warnings

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n;
that it is for first node in FORMER layer
'''

INT = "__import__('code').interact(local=locals())"

@profile
def costfunc(theta, data, ans):
    sig = sigmoid(dot(data, theta))
    return sum(ans * -log(sig) - (1 - ans) * log(1 - sig))

@profile
def costfunc_d(theta, data, ans):
    return dot(data.T, (sigmoid(dot(data, theta)) - ans)) / len(theta)


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


def gradientcheck(annobject, inp, ans):
    '''Relative Difference is consistent!'''
    eps = 1e-10
    grad = []
    for i in range(annobject.totalthetalen):
        annobject.theta[i] += eps
        cost1 = annobject.costfunc(annobject.theta, inp, ans)
        annobject.theta[i] -= 2 * eps
        cost2 = annobject.costfunc(annobject.theta, inp, ans)
        grad.append((cost1 - cost2) / 2 / eps)
        annobject.theta[i] += eps
    gradcalc = a.gradient(a.theta, data, ans)
    grad = array(grad) / len(inp[0])
    print('gradient checking', grad)
    print('calculated gradient', gradcalc)
    print('adiff', grad - gradcalc)
    print('rdiff', grad / gradcalc)
    return isclose(array(grad), gradcalc)


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

    @profile
    def fowardprop(self, allinp, theta=None):
        if theta is None:
            theta = self.theta
        if not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray')
        a = empty((len(allinp), self.unitnum))
        for l in range(self.layercount - 1):
            a[:,self.cumlayernum[l]:self.cumlayernum[l+1]] = allinp
            start, end = self.partialthetalen[l], self.partialthetalen[l+1]
            bias = theta[start:start+self.layernum[l+1]]
            thetaseg = theta[start+self.layernum[l+1]: end].reshape(self.layernum[l], self.layernum[l+1])
            allinp = sigmoid(dot(allinp, thetaseg) + bias)
        a[:,self.cumlayernum[-1]:] = allinp
        return a

    @profile
    def get(self, inp):
        return self.fowardprop(array([inp]))[0, self.cumlayernum[-1]:]  # [0]: first inp's output(while there's only one for .get)

    @profile
    def cost_and_gradient(self, theta, inp, ans):
        a = self.fowardprop(inp, theta)  # stands for activations
        out = a[:, self.cumlayernum[-1]:]
        cost = (ans * -log(out) - (1 - ans) * log(1 - out)).sum()
        g = zeros_like(theta)
        ln = self.layernum
        cln = self.cumlayernum
        fillptr = self.totalthetalen
        lasterror = a[:, self.cumlayernum[-1]:] - ans
        g[fillptr - ln[-2] * ln[-1]: fillptr] = np.inner(a[:, cln[-2]:cln[-1]].T, lasterror.T).flatten()
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
            g[fillptr - ln[l-1] * ln[l]: fillptr] = np.inner(a[:, cln[l-1]: cln[l]].T, lasterror.T).flatten()
            fillptr -= ln[l-1] * ln[l]
            g[fillptr - ln[l]: fillptr] = np.sum(lasterror, axis=0)
            fillptr -= ln[l]
        assert fillptr == 0, 'fillptr should be 0, but is %d' % fillptr
        g /= len(ans)
        return cost, g

    @profile
    def train(self, inp, ans, gtol=1e-5):
        if not (isinstance(inp, ndarray) and isinstance(ans, ndarray)):
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
            minres = minimize(self.cost_and_gradient,
                              self.theta,
                              args=(inp, ans),
                              jac=True,
                              method='BFGS',
                              options={'gtol': gtol})
        self.fpcache_enabled = False
        self.fpcache = None
        self.theta = minres.x
        return minres



def loaddata(setname):
    SUPPORTED_DATASET = ['and', 'or', 'xor', 'nand', 'nor', 'xnor']
    if setname not in SUPPORTED_DATASET:
        raise ValueError('setname is not supported')
    if setname in ['and', 'or', 'xor', 'nand', 'nor', 'xnor']:
        data = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if setname == 'and':
        ans = array([[0], [0], [0], [1]])
    elif setname == 'or':
        ans = array([[0], [1], [1], [1]])
    elif setname == 'xor':
        ans = array([[0], [1], [1], [0]])
    elif setname == 'nand':
        ans = array([[1], [1], [1], [0]])
    elif setname == 'nor':
        ans = array([[1], [0], [0], [0]])
    elif setname == 'xnor':
        ans = array([[1], [0], [0], [1]])
    return data, ans


def handle_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--search', action='store_true', help='Do repetitive search')
    args = parser.parse_args()
    return args.search


if __name__ == '__main__':
    search = handle_args()
    try:
        ANN_DIMENSION = [2, 2, 1]
        data, ans = loaddata('xor')
        if search:
            minval = 100000
            minx = None
            try:
                for i in count():
                    a = ann(ANN_DIMENSION)
                    minres = a.train(data, ans)
                    if minres.fun < minval:
                        print('Minimum value found on %dth attempt: %f' % (i + 1, minres.fun))
                        minval = minres.fun
                        minx = minres.x
                    if i % 10 == 0 and i != 0:
                        print('%d try done..' % (i))
            except KeyboardInterrupt:
                print('Search interrupted with %d try(s) and minimum value of %f found.' % (i, minval))
                if minx is not None:
                    a.theta = minx
                    print('Best theta value is plugged into object a.')
        else:
            print('Module and data loaded.')
    except:
        import traceback
        traceback.print_exc()
    finally:
        __import__('code').interact(local=locals())
