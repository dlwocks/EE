'''
Defines perfectalg, a perfect algorithm for tic-tac-toe
'''
import logging as log
from itertools import product
from random import shuffle, sample

from tttbase import isend, flatten


def donothing(x, _):
    return x


def _permutation(row, r):
    pool = [(row[0], row[1], row[2]), (row[1], row[2], row[0]), (row[0], row[2], row[1])]
    if r:
        shuffle(pool)
    for p in pool:
        yield p


def _row_gen_pos(r=False):
    pool = [[(i, 0), (i, 1), (i, 2)] for i in range(3)] + [[(0, i), (1, i), (2, i)] for i in range(3)] + [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    if r:
        shuffle(pool)
    for p in pool:
        yield p


def _two_in_a_row(board, ainum, r):
    '''
    Return one of positions that completes a two-in-a-row.
    (That is, a position either wins or blocks)
    Returns the one make ainum win if both player has two-in-a-row.
    '''
    temp = None
    for row in _row_gen_pos(r):
        for i, j, k in _permutation(row, r):
            if board[i[0]][i[1]] % 2 == board[j[0]][j[1]] % 2 and board[i[0]][i[1]] != 0 and board[j[0]][j[1]] != 0 and board[k[0]][k[1]] == 0:
                if board[i[0]][i[1]] % 2 == ainum % 2:
                    return k
                else:
                    temp = k
    log.info(repr(temp))
    return temp


def _fork(board, ainum, turn, r):
    tempboard = [[-100 if board[j][i] else 0 for i in range(3)]for j in range(3)]
    for row in _row_gen_pos(r):
        for i, j, k in _permutation(row, r):
            if board[i[0]][i[1]] == board[j[0]][j[1]] == 0 and board[k[0]][k[1]] != 0 and board[k[0]][k[1]] % 2 == ainum % 2:
                tempboard[i[0]][i[1]] += 1
                tempboard[j[0]][j[1]] += 1
                continue
    return [pos for pos in product(range(3), range(3)) if tempboard[pos[0]][pos[1]] >= 2]


def _fork_opponent(board, ainum, turn, r):
    oppofork = _fork(board, (ainum % 2) + 1, turn, r)
    if not oppofork:
        return None
    create_tiar = set()
    for row in _row_gen_pos(r):
        for i, j, k in _permutation(row, r):
            if board[i[0]][i[1]] == board[j[0]][j[1]] == 0 != board[k[0]][k[1]] and board[k[0]][k[1]] % 2 == ainum % 2:
                create_tiar.add((i, j))
    for posone, postwo in create_tiar:
        if posone not in oppofork:
            return postwo
        if postwo not in oppofork:
            return posone
    return oppofork[0]


def perfectalg(board, ainum, turn, rndfrombest=False):
    '''
    Intended to follow Newell and Simon's 1972 tic-tac-toe program's algorithm
    as shown in wikipedia.
    1. Win
    2. Block
    3. Fork
    4. Blocking fork
    5. Center
    6. Opposite Corner (NI)
    7. Empty Corner
    8. Empty Side
    '''
    if turn == 10:
        raise ValueError('Board is full..?')
    pos = _two_in_a_row(board, ainum, rndfrombest)
    if pos:
        log.info('returned in _two_in_a_row')
        return pos
    pos = _fork(board, ainum, turn, rndfrombest)
    if pos:
        log.info('returned in own fork')
        return pos[0]
    pos = _fork_opponent(board, ainum, turn, rndfrombest)
    if pos:
        log.info('returned in opponent fork')
        return pos
    log.info('returned after tiar and fork')
    if board[1][1] == 0:
        return 1, 1
    '''
    if board[0][0] != 0 and board[0][0] % 2 != ainum % 2 and board[2][2] == 0:
        return 2, 2
    if board[2][2] != 0 and board[2][2] % 2 != ainum % 2 and board[0][0] == 0:
        return 0, 0
    if board[0][2] != 0 and board[0][2] % 2 != ainum % 2 and board[2][0] == 0:
        return 2, 0
    if board[2][0] != 0 and board[2][0] % 2 != ainum % 2 and board[0][2] == 0:
        return 0, 2
    '''
    if rndfrombest:
        f = sample
    else:
        f = donothing
    for i in f([0, 2], 2):
        for j in f([0, 2], 2):
            if board[i][j] == 0:
                return i, j
    for i, j in f([(0, 1), (1, 0), (1, 2), (2, 1)], 4):
        if board[i][j] == 0:
            return i, j
    assert False, board


def perfectalg_r(board, ainum, step):
    return perfectalg(board, ainum, step, rndfrombest=True)

class Abpai():
    def __init__(self):
        self._outi = -1
        self._outj = -1
        self.posls = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]

    def __call__(self, board, ainum, step):
        self.abpalg(board, float('-inf'), float('inf'), step, isbase=True)
        return self._outi, self._outj

    def abpalg(self, board, alpha, beta, step, isbase=False):
        end = isend(board, step)
        if end is not None:
            return end
        for i, j in self.posls:
            if board[i][j] != 0:
                continue
            board[i][j] = step
            point = self.abpalg(board, alpha, beta, step + 1)
            board[i][j] = 0
            if step % 2: # ismax
                if point > alpha:
                    alpha = point
                    if isbase:
                        self._outi = i
                        self._outj = j
                if alpha >= beta:
                    break
            else:
                if point < beta:
                    beta = point
                    if isbase:
                        self._outi = i
                        self._outj = j
                if alpha >= beta:
                    break
        if step % 2:
            return alpha
        else:
            return beta


def label(board, ainum=None, step=None):
    if step is None:
        step = max(flatten(board)) + 1
    ai = Abpai()
    return ai.abpalg(board, float('-inf'), float('inf'), step)
