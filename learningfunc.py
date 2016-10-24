from numpy import array, dot, log, e, ndarray, append, sqrt
from random import random, uniform
from copy import copy
from itertools import chain
from functools import reduce
from scipy.optimize import minimize

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n;
that it is for first node in FORMER layer
'''


def debug(message, param):
    ALLOWED_MSG_LIST = ['out']
    if message in ALLOWED_MSG_LIST:
        print(message + ': ' + str(param))


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


def _rndinit(layernum):  # totalthetalen is solely for debug purpose
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


def _regseg(theta, layernum, partialthetalen):
    ret = array([])
    temp = None
    for l, p in zip(layernum, partialthetalen):
        if temp is not None:
            bias = array([0 for i in range(l)])
            thetaseg = append(bias, theta[temp+l:p])
            ret = append(ret, thetaseg)
        temp = p
    return ret


class ann(object):
    def __init__(self, layernum, theta=None, reg=0.001):
        if not isinstance(layernum, list):
            raise TypeError('param layernum is not list')
        if len(layernum) < 2:
            raise ValueError('param layernum is too short.'
                             'The network should at least consist input/output layer')
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
        self.reg = reg

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
                raise RuntimeError('input size doesn\'t match')
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
        return self.fowardprop([inp], return_out=True, return_a=a)

    def costfunc(self, theta, inp, ans):
        out = array(self.fowardprop(inp, theta, return_out=True))
        costlist = [-log(tout)[0] if tans == 1 else -log(1 - tout)[0]
                    for tout, tans in zip(out, ans)]
        # FIXME: -log(tout)**[0]** only works with ann of 1 output
        # FIXME: tans: only considered binary ans
        # costlist = sum(ans * -log(out).T - (1 - ans) * log(1 - out).T)
        debug('out', out)
        debug('ans', ans)
        debug('-log(out).T', -log(out).T)
        debug('-log(1 - out).T', -log(1 - out).T)
        debug('cost', costlist)
        totalcost = sum(costlist) / len(ans)
        regtheta = _regseg(theta, self.layernum, self.partialthetalen)
        debug('theta', theta)
        debug('regtheta', regtheta)
        regcost = sum([i**2 for i in regtheta]) / len(regtheta) / 2 * self.reg  # XXX: len(regtheta) or len(ans)?
        debug('totalcost', totalcost)
        debug('regcost', regcost)
        return totalcost + regcost

    def gradient_single(self, theta, inp, ans):
        inp = array([inp])
        a = self.fowardprop(inp, theta, return_a=True)[0]
        lasterror = a[-1] - ans
        debug('a', a)
        debug('lasterror', lasterror)
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
        debug('gradient_single', delta)
        return array(delta)

    def gradient(self, theta, inp, ans):
        #  Parallel Delta Version:
        #  g = [i / len(ans) for i in reduce(lambda a, b: a + b, [self.gradient_single(theta, thisinp, thisans) for thisinp, thisans in zip(inp, ans)])]
        #  Series Delta Version:
        init_theta = copy(theta)
        for thisinp, thisans in zip(inp, ans):
            grad = self.gradient_single(theta, thisinp, thisans) / len(ans)
            theta = theta + grad
        g = theta - init_theta
        regtheta = _regseg(theta, self.layernum, self.partialthetalen)
        reggrad = regtheta / len(regtheta) * self.reg
        debug('reggrad', reggrad)
        debug('gradient', g)
        return g + reggrad

    def train(self, inp, ans):
        minres = minimize(self.costfunc,
                          self.theta,
                          args=(inp, ans),
                          jac=self.gradient,
                          method='BFGS')
        self.theta = minres.x
        print(minres)

if __name__ == '__main__':
    try:
        a = ann([2, 2, 1])
        data = array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
        ans = array([0, 1, 1, 0])  # FIXME: General implementation should work in ans = array([[0], [1], [1], [0]]), but it is not working
        subdata = array([[0, 0],
                         [1, 1],
                         [1, 0]])
        subans = array([0, 0, 1])
        a.train(data, ans)
    except:
        import traceback
        traceback.print_exc()
    finally:
        __import__('code').interact(local=locals())
