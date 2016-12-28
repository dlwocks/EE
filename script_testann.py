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
a = ann([1, 4, 1])
a.train(inp, ans)
'''
timeit(code,setup=setup, number=100)

# Commit 177557785f620ef51d7671e9434d5af6d63ed999: cythonann takes 20.s


####################################################################################

# The following script is to help you determine whether a particular ann implementation is bahaving correctly.
# import ann as an ann class that you want to test before this line (say, from cythonann import ann), and copy following line

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from cythonann import ann

from numpy import sin, abs, linspace, arange, array
def target(inp):
    return abs(sin(inp) /  inp)

inp = linspace(-10, 10, 1000)[None].T
ans = target(inp)

a = ann([1, 20, 1])
minres = a.train(inp, ans)  # This probably needs a few minute
print('fun:', minres.fun)
x = arange(-10, 10, 0.01)
import matplotlib.pyplot as plt
plt.plot(x, target(x))
plt.plot(x, [a.get(array([i])) for i in x])
plt.show()

# Or more conveniently using function call:
def test(layernum=[1, 10, 1]):
    def mincost(ans):
        from numpy import log
        return (ans * -log(ans) - (1 - ans) * log(1 - ans)).sum()
    a = ann(layernum)
    minres = a.train(inp, ans)
    print('fun(mincost deducted):', minres.fun - mincost(ans))
    x = arange(-10, 10, 0.01)
    plt.plot(x, target(x))
    plt.plot(x, [a.get(array([i])) for i in x])
    print('\a')
    plt.show()


# Find out the least possible cost for the given answer:
from numpy import log
(ans * -log(ans) - (1 - ans) * log(1 - ans)).sum()