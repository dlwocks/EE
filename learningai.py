from numpy import array, dot, log, e
from scipy.optimize import minimize
from copy import copy
from itertools import repeat, count
from random import random, randint
import matplotlib.pyplot as plt

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

    def _getdiff(self, arr):
        for i in range(len(arr)-1, 0, -1):
            arr[i] = arr[i] - arr[i-1]
        return arr

    def _checkdiv(self, arr, difftol):
        conv = True
        largest = 0
        for f in range(self.VAL_FEATURE_NUM):
            diff = abs(arr[-1][f] - arr[-2][f])
            if diff > difftol:
                conv = False
                if diff > largest:
                    largest = diff
        if conv:
            return 0
        else:
            return largest

    def startlearn(self, game='converge', difftol=0.01, opponent='random'):
        if game == 'converge':
            iterator = count()
            THETA_CHECK_STEP = 10
        else:
            iterator = range(game)
            THETA_CHECK_STEP = game // 100
        thetas = []
        for c in iterator:
            if c % THETA_CHECK_STEP == 0:
                if c != 0:
                    if game == 'converge' and c >= THETA_CHECK_STEP * 2:
                        div = self._checkdiv(thetas, difftol)
                        if not div:
                            print('%d games done: Theta value has successfully converged.' % c)
                            break
                        else:
                            print('%d games done: Theta value is yet to converge. Largest divergence: %f' % (c, div))
                    else:
                        print('%d games done..' % c)
                thetas.append(self.theta_value)
            board = [[0 for i in range(3)]for i in range(3)]
            ainum = randint(1, 2)
            end = 0
            step = 1
            while not end and step < 10:
                if step % 2 == ainum % 2:
                    i, j = self.getstep(board, ainum, step)
                else:
                    if opponent == 'self':
                        i, j = self.getstep(board, ainum, step)
                    elif opponent == 'random':
                        i, j = self._randomstep(board)
                    else:
                        assert False, 'param opponent is not self or random.'
                board[i][j] = step
                end = isend(board)
                if end == 1 or end == 2:
                    self.train_value(board, end)
                    break
                step += 1
        thetas.append(self.theta_value)
        print('learning successfully terminated with %d game(s) done.'
              'Final theta value:\n%s' % (c, repr(self.theta_value)))
        thetas = array(thetas).T
        for i in range(9):
            plt.plot([i for i in range(0, c+1, THETA_CHECK_STEP)], thetas[i])
        plt.show()
        plt.close('all')
        for i in range(9):
            plt.plot([i for i in range(0, c+1, THETA_CHECK_STEP)], self._getdiff(thetas[i]))
        plt.plot([i for i in range(0, c+1, THETA_CHECK_STEP)], [0 for i in range(len(thetas[0]))], linewidth=2.0, color='black')
        plt.show()
        return self

if __name__ == '__main__':
    logreg_ai().startlearn()
