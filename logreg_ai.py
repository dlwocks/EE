from numpy import array, dot, log, e, ndarray
from scipy.optimize import minimize
from copy import copy
from itertools import repeat, count
from random import random, randint
import matplotlib.pyplot as plt
import warnings

from learningfunc import costfunc, costfunc_d
from ttthelper import isend, emptyspace_pos
from base_ai import base_ai


class logreg_ai(base_ai):
    def __init__(self, t=None, feature=['board']):
        self.data = []
        self.ans = []
        if not feature:
            raise ValueError('no feature is given')
        for f in feature:
            if f not in self.FEATURE_FUNC_MAP.keys():
                raise ValueError('The feature "%s" is not supported' % f)
        self.feature = feature
        if t is None:
            self.theta_value = array([(random()-0.5)/1000 for _ in range(self.feature_num)])
        else:
            self.theta_value = t

    def _add(self, board, end):
        feature = self.featureize_in_piece(board)
        self.data.extend(feature)
        self.ans.extend(list(repeat(end, len(feature))))

    def _train_value(self, board, end):
        assert end in [1, 2, 0.5]
        if end == 2:
            end = 0
        self._add(board, end)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.theta_value = minimize(costfunc, self.theta_value, args=(array(self.data), array(self.ans)), jac=costfunc_d, method='BFGS').x

    def getstep(self, board, ainum, step):
        mi, mj, mdot = 0, 0, -10000 if ainum % 2 else 10000
        for nextboard, i, j in emptyspace_pos(board, step):
            nextboard = self.featureize_final(nextboard)
            dotval = dot(nextboard, self.theta_value)
            if ainum % 2 == 1 and dotval > mdot:
                mi, mj, mdot = i, j, dotval
            elif ainum % 2 == 0 and dotval < mdot:
                mi, mj, mdot = i, j, dotval
        return mi, mj

    def _randomstep(self, board):
        while True:
            i, j = randint(0, 2), randint(0, 2)
            if board[i][j] == 0:
                return i, j

    def _getdiff(self, arr):
        for i in range(len(arr)-1, 0, -1):
            arr[i] = arr[i] - arr[i-1]
        return arr

    def _checkdiv(self, arr, difftol):
        conv = True
        largest = 0
        for f in range(self.feature_num):
            diff = abs(arr[-1][f] - arr[-2][f])
            if diff > difftol:
                conv = False
                if diff > largest:
                    largest = diff
        if conv:
            return 0
        else:
            return largest

    def startlearn(self, game='converge', difftol=0.01, opponent='random', graph=True, pt=True):
        if opponent not in ['self', 'random']:
            raise ValueError('param opponent is not self or random.')
        if game == 'converge':
            iterator = count()
            THETA_CHECK_STEP = 10
        else:
            iterator = range(game)
            THETA_CHECK_STEP = game // 100
        theta_rec = []
        for c in iterator:
            if c % THETA_CHECK_STEP == 0:
                if c != 0:
                    if game == 'converge' and c >= THETA_CHECK_STEP * 2:
                        div = self._checkdiv(theta_rec, difftol)
                        if not div:
                            if pt:
                                print('%d games done: Theta value has successfully converged.' % c)
                            c -= 1
                            break
                        elif pt:
                            print('%d games done: Theta value is yet to converge. Largest divergence: %f' % (c, div))
                    elif pt:
                        print('%d games done..' % c)
                theta_rec.append(self.theta_value)
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
                assert board[i][j] == 0
                board[i][j] = step
                if 9 >= step >= 5:
                    end = isend(board, step+1)
                    if end:
                        self._train_value(board, end)
                        break
                step += 1
        theta_rec.append(self.theta_value)
        if pt:
            print('learning successfully terminated with %d game(s) done.'
                  'Final theta value:\n%s' % (c+1, repr(self.theta_value)))
        if graph:
            theta_rec = array(theta_rec).T
            for i in range(9):
                plt.plot([i for i in range(0, c+2, THETA_CHECK_STEP)], theta_rec[i])
            plt.show()
            plt.close('all')
            for i in range(9):
                plt.plot([i for i in range(0, c+2, THETA_CHECK_STEP)], self._getdiff(theta_rec[i]))
            plt.plot([i for i in range(0, c+2, THETA_CHECK_STEP)], [0 for i in range(len(theta_rec[0]))], linewidth=2.0, color='black')
            plt.show()
        return self

if __name__ == '__main__':
    logreg_ai().startlearn()
