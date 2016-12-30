# The following script is to help you determine the general speed of a particular ann implementation.
# This script does NOT test the accuracy of the implementation.
# import ann as an ann class that you want to test before this line (say, from cythonann import ann), and copy following line
#
# Note on how to profile: go to site-packages/line-profiler, and run py -3 kernprof.py -l [script file]
from timeit import timeit
setup = '''
from __main__ import ann
from numpy import array, sin, abs
def target(inp):
    return abs(sin(4 * inp) / 4 / inp)

inp = array([[(i+0.1)/10] for i in range(8)])
ans = target(inp)
'''
code = '''
a = ann([1, 2, 1])
a.train(inp, ans)
'''
timeit(code,setup=setup, number=1)

# Commit 177557785f620ef51d7671e9434d5af6d63ed999: cythonann takes 20.s


####################################################################################

# The following script is to help you determine whether a particular ann implementation is bahaving correctly.
# import ann as an ann class that you want to test before this line (say, from cythonann import ann), and copy following line

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from cythonann import ann

from numpy import sin, abs, linspace, arange, array
# Target1
def target(inp):
    return abs(sin(inp) /  inp)

inp = linspace(-10, 10, 1000)[None].T
# Target2
def target(inp):
    return sin(inp)

inp = linspace(0.01, 3.14, 300)[None].T


ans = target(inp)

a = ann([1, 20, 1])
minres = a.train(inp, ans)  # This probably needs a few minute
print('fun:', minres.fun)
x = linspace(0.01, 3.14, 300)
import matplotlib.pyplot as plt
plt.plot(x, target(x))
plt.plot(x, [a.get(array([i])) for i in x])
plt.show()


# Or more conveniently using function call:
def test(target=None, hiddenlayer=[10], plot=True, rng=(-10,10), density=50, verbose=False):
    num = round((rng[1] - rng[0]) * density)
    if target is None:
        try:
            target = globals()['target']
        except KeyError:
            raise RuntimeError('Could not find target function!')
    try:  # finding cached inp and ans
        if __rng == rng and __density == density and target is __target:
            inp = __inp
            ans = __ans
        else:
            raise NameError
    except NameError:
        inp = linspace(rng[0], rng[1], num)[None].T
        ans = target(inp)
        globals()['__inp'] = inp
        globals()['__ans'] = ans
        globals()['__rng'] = rng
        globals()['__density'] = density
        globals()['__target'] = target
    if len(ans) != num:
        raise RuntimeError('target function did not return array of same length with input.')
    import matplotlib.pyplot as plt
    def mincost(ans):
        from numpy import log
        return (ans * -log(ans) - (1 - ans) * log(1 - ans)).sum()
    a = ann([1] + hiddenlayer + [1])
    minres = a.train(inp, ans)
    if plot:
        plt.plot(inp, target(x))
        plt.plot(inp, [a.get(array([i])) for i in x])
        plt.show()
    return minres.fun - mincost(ans)


# Find out the least possible cost for the given answer:
from numpy import log
(ans * -log(ans) - (1 - ans) * log(1 - ans)).sum()

# Test the relation of hidden layer to convergence of ann, using the above test function
from statistics import mean
import numpy as np
ls = []
for i in range(1, 51):
    c = min([test([i], plot=False) for t in range(3)])
    ls.append(c)
    print('%d hidden layer tested' % i)

import matplotlib.pyplot as plt
x = np.arange(1, 51)
plt.plot(x, ls)
plt.show()
