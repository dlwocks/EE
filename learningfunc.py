from numpy import array, dot, log, ndarray, append, sqrt, array_equal, zeros_like, isclose
from random import uniform
from copy import copy
from itertools import count
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
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



'''
TODO:
Reinforcement learning using trained value network
Policy Network Training using perfectdataset
'''
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
            # self.theta = array([0 for i in range(self.totalthetalen)])
        self.fpcache_enabled = False
        self.fpcache_theta = None
        self.fpcache = None

    #@profile
    def fowardprop(self, allinp, theta=None, return_out=False):
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
                inp = append(array([1]), inp)  # Add bias unit
                temp_a.append(inp)  # Append bias_added layer to temp_a
                start, end = self.partialthetalen[l], self.partialthetalen[l+1]
                thetaseg = theta[start: end]
                inp = sigmoid(dot(inp, thetaseg.reshape(self.layernum[l]+1, self.layernum[l+1])))
            temp_a.append(inp)
            a.append(temp_a)
        if self.fpcache_enabled:
            self.fpcache_theta = copy(theta)
            self.fpcache = a
        return a

    def get(self, inp):
        return self.fowardprop(array([inp]))[0][-1]  # [0]: first inp's output(while there's only one)

    #@profile
    def costfunc(self, theta, inp, ans):
        out = array(self.fowardprop(inp, theta)).T[-1]
        totalcost = 0
        for tout, tans in zip(out, ans):
            totalcost += sum(tans * -log(tout) - (1 - tans) * log(1 - tout))
        return totalcost

    #@profile
    def gradient_single(self, theta, a, ans):
        lasterror = a[-1] - ans
        delta = list((a[-2][None].T * lasterror[None]).flatten())
        for i in range(self.layercount - 2, 0, -1):
            start, end = self.partialthetalen[i], self.partialthetalen[i+1]
            thetaseg = theta[start: end].reshape(
                self.layernum[i]+1, self.layernum[i+1])
            d = dot(thetaseg[1:], lasterror)
            agrad = (a[i][1:] * (1 - a[i][1:]))
            lasterror = d * agrad  # This is in fact this(ith) layer's error; below same.
            subdelta = list((a[i-1][None].T * lasterror[None]).flatten())
            delta = subdelta + delta
        return array(delta)

    #@profile
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

    #@profile
    def train(self, inp, ans):
        if not (isinstance(inp, ndarray) and isinstance(ans, ndarray)):
            raise TypeError
        if not (len(inp.shape) == 2 and len(ans.shape) == 2 and len(inp) == len(ans) and len(inp[0]) == self.layernum[0] and len(ans[0] == self.layernum[-1])):
            raise ValueError((len(inp.shape) == 2, len(ans.shape) == 2, len(inp) == len(ans), len(inp[0]) == self.layernum[0]))
        self.fpcache_enabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minres = minimize(self.costfunc,
                              self.theta,
                              args=(inp, ans),
                              jac=self.gradient,
                              method='BFGS')
        self.fpcache_enabled = False
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
