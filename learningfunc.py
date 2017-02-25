from numpy import dot, log, sqrt
import numpy as np
from random import uniform
from itertools import accumulate
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

'''
Important definition:
-In an thetaseg, the first n theta represents 11, 12, 13, ..., 1n;
that it is for first node in FORMER layer
'''
def costfunc(theta, data, ans):
    sig = sigmoid(dot(data, theta))
    return sum(ans * -log(sig) - (1 - ans) * log(1 - sig))

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
    inittheta = np.array([])
    for outnum in layernum:
        if inpnum:
            eps_init = sqrt(6)/sqrt(inpnum+outnum)
            appendedtheta = np.array([uniform(-eps_init, eps_init)
                                   for i in range((inpnum+1)*outnum)])
            inittheta = np.append(inittheta, appendedtheta)
        inpnum = outnum
    return inittheta


NEAR_ZERO = np.nextafter(0, 1)
NEAR_ONE = np.nextafter(1, -1)

class ann(object):
    def __init__(self, layernum, theta=None, reg=0):
        '''Initializes the ANN class.

        Args:
            layernum: A list. Its length would be the number of layer of ANN,
                nth number in the list would be the number of neuron in corresponding layer.
            theta: Optional. If given, an np.ndarray. The weight(theta) of ANN is initialized with the value.
                If not given, theta is initialized randomly.
            reg: Optional. If given, an integer. Specifies the regularization parameter for the ANN.
                If not given, reg=0(regularization is not used).
        Returns:
            The initialized instance of class ann.
        '''
        if not isinstance(layernum, list):
            raise TypeError('param layernum is not list')
        if len(layernum) < 2:
            raise ValueError('param layernum is too small.'
                             'It should at least consist input/output layer')
        self.reg = reg
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

    def _regtheta(self):
        '''Finds the segment of theta to be regularized, leaving theta of bias term as zero.
        
        Args:
            None.
        Returns:
            The self.theta where theta connected to bias term is changed to zero.
        '''
        ret = np.zeros_like(self.theta)
        temp = 0
        for i, l in enumerate(self.partialthetalen[1:]):
            ret[temp+self.layernum[i+1]: l] = self.theta[temp+self.layernum[i+1]: l]
            temp = l
        return ret

    def fowardprop(self, allinp, theta=None):
        '''Executes forward propagation. 

        Args: 
            allinp: A 2-dimensional np.ndarray. The array of input vectors.
            theta: A 1-dimensional np.ndarray. The weight of the network. If not given, assumed self.theta.
        Returns:
            A 2-dimensional np.ndarray, with each row representing all activation in all layer from the corresponding input.
        '''
        if theta is None:
            theta = self.theta
        if not isinstance(allinp, np.ndarray):
            raise TypeError('input is not a ndarray')
        a = np.empty((len(allinp), self.unitnum))
        for l in range(self.layercount - 1):
            a[:, self.cumlayernum[l]:self.cumlayernum[l+1]] = allinp
            start, end = self.partialthetalen[l], self.partialthetalen[l+1]
            bias = theta[start:start+self.layernum[l+1]]
            thetaseg = theta[start+self.layernum[l+1]: end].reshape(self.layernum[l], self.layernum[l+1])
            allinp = sigmoid(dot(allinp, thetaseg) + bias)
        a[:, self.cumlayernum[-1]:] = allinp
        return a

    def get(self, inp):
        '''Get forward propagation result for a single input.
        
        Args:
            inp: A 1-dimensional array-like object. The input given.
        Returns:
            A 1-dimensional np.ndarray. The output of the network from the input.
        '''
        return self.fowardprop(np.array([inp]))[0, self.cumlayernum[-1]:]  # [0]: first inp's output(while there's only one for .get)

    def costfunc(self, inp, ans):
        ''' The cost that the network would get from the given set of input and target output.

        Args:
            inp: A 2-dimensional np.ndarray. The input.
            ans: A 2-dimensional np.ndarray. The target output.
        Return:
            A float. The cost that the network would get from the given set of input and target output.
        '''
        assert len(inp) == len(ans)
        out = self.fowardprop(inp, self.theta)[:, self.cumlayernum[-1]:]
        np.place(out, out < NEAR_ZERO, NEAR_ZERO)
        np.place(out, out > NEAR_ONE, NEAR_ONE)  # Avoid overflow in log
        cost = (ans * -np.log(out) - (1 - ans) * np.log(1 - out)).sum()
        if self.reg:
            cost += self.reg * (self._regtheta()**2).sum() / 2
        cost /= len(ans)
        return cost

    def cost_and_gradient(self, theta, inp, ans):
        '''The cost and the partial derivative of cost with respect to each weight for the given set  of input and target output.

        Args:
            theta: A 1-dimensional np.ndarray. The current weight of the network.
            inp: A 2-dimensional np.ndarray. The input.
            ans: A 2-dimensional np.ndarray. The target output.
        Return:
            A tuple. First item is the cost, second item is a list of partial derivative(gradient).
        '''
        a = self.fowardprop(inp, theta)  # stands for activations
        out = a[:, self.cumlayernum[-1]:]
        np.place(out, out < NEAR_ZERO, NEAR_ZERO)
        np.place(out, out > NEAR_ONE, NEAR_ONE)  # Avoid overflow in log
        cost = (ans * -log(out) - (1 - ans) * log(1 - out)).sum()
        if self.reg:
            cost += self.reg * (self._regtheta()**2).sum() / 2
        g = np.zeros_like(theta)
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
            aseg = a[:, cln[l]: cln[l+1]]
            agrad = aseg * (1 - aseg)
            lasterror = d * agrad
            g[fillptr - ln[l-1] * ln[l]: fillptr] = np.inner(a[:, cln[l-1]: cln[l]].T, lasterror.T).flatten()
            fillptr -= ln[l-1] * ln[l]
            g[fillptr - ln[l]: fillptr] = np.sum(lasterror, axis=0)
            fillptr -= ln[l]
        assert fillptr == 0, 'fillptr should be 0, but is %d' % fillptr
        if self.reg:
            g += self.reg * self._regtheta()
        cost /= len(ans)
        g /= len(ans)
        return cost, g

    def train(self, inp, ans, gtol=1e-5):
        '''Minimizes the error on given input(inp) and corresponding target output(ans).

        Args:
            inp: A 2-dimensional np.ndarray. The input to neural network.
            ans: A 2-dimensional np.ndarray. The corresponding target output of the given input.
            gtol: Optional. If given, a float. gradient tolerance in BFGS method. If not given, assumed 1e-5.
        Return:
            Minimization Result, which is returned in scipy.optimize.minimize.
        '''
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
        minres = minimize(self.cost_and_gradient,
                          self.theta,
                          args=(inp, ans),
                          jac=True,
                          method='BFGS',
                          options={'gtol': gtol})
        self.theta = minres.x
        return minres
