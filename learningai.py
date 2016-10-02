from numpy import array, dot, log, e
from scipy.optimize import minimize
from copy import copy
from itertools import repeat
from random import random, randint

from tttlearning import costfunc, costfunc_d, sigmoid
from ttttester import isend


class logreg_ai(object):
    VAL_FEATURE_NUM = 9
    data = []
    ans = []

    def __init__(self, t=array([0 for _ in range(VAL_FEATURE_NUM)])):
        self.theta_value = t

    def _emptyspace_pos(self, board, step):
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = step
                    yield board, i, j
                    board[i][j] = 0

    def _add(self, board, end):
        board = list(array(board).reshape((9,)))
        ar = [0 for i in range(9)]
        for i in range(1, 10):
            try:
                ar[board.index(i)] = 1 if i % 2 else -1
            except ValueError:
                i -= 1
                break
            self.data.append(copy(ar))
        self.ans.extend(list(repeat(end, i)))

    def train_value(self, board, end):
        assert end == 1 or end == 2
        end = end % 2
        self._add(board, end)
        self.theta_value = minimize(costfunc, self.theta_value, args=(array(self.data), array(self.ans)), jac=costfunc_d, method='BFGS').x

    def getstep(self, board, ainum, step):
        mi, mj, maxdot = 0, 0, -10000
        for nextboard, i, j in self._emptyspace_pos(board, step):
            nextboard = array(nextboard).reshape((9,))
            dotval = dot(nextboard, self.theta_value)
            if dotval > maxdot:
                mi, mj, maxdot = i, j, dotval
        return mi, mj

    def _randomstep(self, board):
        count = 0
        while True:
            i, j = randint(0, 2), randint(0, 2)
            if board[i][j] == 0:
                return i, j
            count += 1
            assert count <= 10000

    def startlearn(self, game=1000):
        for _ in range(game):
            if _ > 0 and _ % (game//50) == 0:
                print('%d games done..\nCurrent theta value:\n%s' % (_, self.theta_value))
            try:
                board = [[0 for i in range(3)]for i in range(3)]
                ainum = randint(1, 2)
                end = 0
                step = 1
                while not end and step < 10:
                    if step % 2 == ainum % 2:
                        i, j = self.getstep(board, ainum, step)
                    else:
                        i, j = self._randomstep(board)
                    board[i][j] = step
                    end = isend(board)
                    if end == 1 or end == 2:
                        self.train_value(board, end)
                        break
                    step += 1
            except:
                print('An unexpected error occured in game %d.' % _)
                assert False
        print('learning successfully terminated with %d game(s) done.'
              'Final theta value:\n%s' % (game, repr(self.theta_value)))
        return self
