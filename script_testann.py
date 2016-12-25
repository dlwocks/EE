# The following script is to help you determine the general speed of a particular ann implementation.
# This script does NOT test the accuracy of the implementation.
# import ann as an ann class that you want to test before this line (say, from cythonann import ann), and copy following line
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
# After fixing costfunc(not using tans, tout): 18.s
# After fixing gradient_single(filling in pre-set delta): 15.s
# After fixing fowardprop(use array for a in fowardprop): 14.4s

####################################################################################

# The following script is to help you determine whether a particular ann implementation is bahaving correctly.
# import ann as an ann class that you want to test before this line (say, from cythonann import ann), and copy following line

import matplotlib.pyplot as plt
from numpy import array, sin, abs, arange
from random import uniform
def target(inp):
    return abs(sin(inp) /  inp)

inp = array([[uniform(-10,10)] for i in range(1000)])
ans = target(inp)
a = ann([1, 20, 1])
a.train(inp, ans)  # This probably needs a few minute
x = arange(-10, 10, 0.01)
plt.plot(x, target(x))
plt.plot(x, [a.get(array([i])) for i in x])
plt.show()
