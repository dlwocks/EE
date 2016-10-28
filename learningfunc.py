from numpy import array, dot, log, e, ndarray, append, sqrt
from random import random, uniform
from copy import copy
from itertools import chain, count
from functools import reduce
from scipy.optimize import minimize
import warnings

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n;
that it is for first node in FORMER layer
'''


def debug(message, param, always=False):
    ALLOWED_MSG_LIST = []
    if message in ALLOWED_MSG_LIST or always:
        print(message + ': ' + str(param))


def sigmoid(z):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = 1/(1+e**(-z))
    return ret


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
            # self.theta = array([(random()-0.5)
            #                     for i in range(self.totalthetalen)])
            # self.theta = array([0 for i in range(self.totalthetalen)])

    def fowardprop(self, allinp, theta=None, return_a=False, return_out=False):
        if theta is None:
            theta = self.theta
        if isinstance(allinp, list):
            allinp = array(allinp)
        elif not isinstance(allinp, ndarray):
            raise TypeError('input is not a ndarray or a list')
        if return_a:
            a = []
        if return_out:
            out = []
        for inp in allinp:
            temp_a = [append(array([1]), inp)]
            if len(inp) != self.layernum[0]:
                raise RuntimeError('input size doesn\'t match. length of input is %d, while it should be %d' % (len(inp), self.layernum[0]))
            for l in range(self.layercount - 1):
                inp = append(array([1]), inp)  # Add bias unit
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                thetaseg = theta[start: end]
                inp = _fowardprop(thetaseg, inp)
                if return_a:
                    if l == self.layercount - 2:
                        temp_a.append(inp)
                    else:
                        temp_a.append(append(array([1]), inp))
            if return_a:
                a.append(temp_a)
            if return_out:
                # HACK: Avoid overflow in log
                for i, o in enumerate(inp):
                    if o == 1:
                        inp[i] = 1 - 1e-10
                    elif o == 0:
                        inp[i] = 1e-10
                out.append(inp)
        if return_a and return_out:
            return out, a
        elif return_out:
            return out
        elif return_a:
            return a
        else:
            raise RuntimeError('fowardprop call without expecting any return')

    def get(self, inp, a=False):
        return self.fowardprop(array([inp]), return_out=True, return_a=a)[0]  # [0]: first inp's output(while there's only one)

    def costfunc_single(self, theta, out, ans):
        c = sum(ans * -log(out) - (1 - ans) * log(1 - out))
        assert c >= 0, (out, ans)
        return c

    def costfunc(self, theta, inp, ans):
        out = array(self.fowardprop(inp, theta, return_out=True))
        costlist = [self.costfunc_single(theta, thisout, thisans) for thisout, thisans in zip(out, ans)]
        totalcost = sum(costlist)
        return totalcost

    def gradient_single(self, theta, inp, ans):
        inp = array([inp])
        a = self.fowardprop(inp, theta, return_a=True)[0]
        lasterror = a[-1] - ans
        delta = list(chain.from_iterable(a[-2][None].T * lasterror[None]))
        for i in range(self.layercount - 2, 0, -1):
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = theta[start: end].reshape(
                self.layernum[i]+1, self.layernum[i+1])
            d = dot(thetaseg[1:], lasterror)
            agrad = (a[i][1:] * (1 - a[i][1:]))
            thiserror = d * agrad
            lasterror = thiserror
            delta = list(chain.from_iterable(a[i-1][None].T * lasterror[None])) + delta
        return array(delta)

    def gradient(self, theta, inp, ans):
        PARALLEL = True
        if PARALLEL:
            g = [i / len(ans) for i in reduce(lambda a, b: a + b, [self.gradient_single(theta, thisinp, thisans) for thisinp, thisans in zip(inp, ans)])]
        else:  # Series Delta
            init_theta = copy(theta)
            for thisinp, thisans in zip(inp, ans):
                grad = self.gradient_single(theta, thisinp, thisans) / len(ans)
                theta = theta + grad
            g = theta - init_theta
        return array(g)

    def train(self, inp, ans):
        if not (isinstance(inp, ndarray) and isinstance(ans, ndarray)):
            raise TypeError
        if not (len(inp.shape) == 2 and len(ans.shape) == 2 and len(inp) == len(ans) and len(inp[0]) == self.layernum[0] and len(ans[0] == self.layernum[-1])):
            raise ValueError((len(inp.shape) == 2, len(ans.shape) == 2, len(inp) == len(ans), len(inp[0]) == self.layernum[0]))
        minres = minimize(self.costfunc,
                          self.theta,
                          args=(inp, ans),
                          jac=self.gradient,
                          method='BFGS')
        self.theta = minres.x
        debug('minres', minres)
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
        data, ans = loaddata('xnor')
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
