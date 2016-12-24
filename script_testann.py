# This is a script to help you determine the general speed of a particular ann implementation.
# This script does NOT test the accuracy of the implementation.
# import ann as an ann class before this line (say, from cythonann import ann), and copy following line
from timeit import timeit
setup = '''
from __main__ import ann
from numpy import array, sin, abs
from random import uniform
def target(inp):
    return abs(sin(4 * inp) / 4 / inp)
inp = array([[(i+0.1)/10] for i in range(5)])
ans = target(inp)
'''
code = '''
a = ann([1, 4, 1])
a.train(inp, ans)
'''
timeit(code,setup=setup, number=100)
