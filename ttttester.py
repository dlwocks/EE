'''Tic-Tac-Toe tester.
The board in this system is a 3*3 array.
0 in the array represent empty space.
Odd number in the array represents 'odd player', who definitely goes first.
Even number in the array represents 'even player', who definitely goes second.
'''
import logging as log
from itertools import product, repeat
from numpy import array
from copy import copy, deepcopy
from random import randint

from ai import algorithm_wiki

global count

log.basicConfig(filename='ttttester.log',
                filemode='w',
                level=log.INFO,
                format='%(asctime)s %(message)s')
RETURNBEFORE = False
count = {'AI': 0, 'Iterator': 0, 'Draw': 0}


def printboard(board, signmap={0: "+", 1: "O", 2: "X"}, hspace='-'):
    '''
    Board printer.
    '''
    ret = '\n'
    for y, x in product(range(len(board)), range(len(board[0]))):
        ret += str(signmap[board[y][x]]) if signmap else str(board[y][x])
        ret += '\n' if x == len(board[0])-1 else hspace
    return ret


def _isend(nums):
    """
    Return 1 if the odd player wins by this set.
    Return 2 if the even player wins by this set.
    Return 0 elsewise.
    """
    num1, num2, num3 = nums
    if num1 == 0 or num2 == 0 or num3 == 0:
        return 0
    if num1 % 2 == num2 % 2 == num3 % 2:
        return 1 if num1 % 2 else 2
    return 0


def isend(board, turn=None):
    """
    Return 1 if the odd player wins the game.
    Return 2 if the even player wins the game.
    Return None elsewise (implicitly)
    """
    for end in map(_isend, _row_gen(board)):
        if end:
            return end
    if turn == 10:
        return 0.5


def _row_gen(board):
    '''
    There are 8 ways to end a tic-tac-toe:
    horiziontal 3, vertical 3, diagonal 2.
    This generates every condition.
    '''
    for i in range(3):
        yield [board[i][0], board[i][1], board[i][2]]
        yield [board[0][i], board[1][i], board[2][i]]
    yield[board[0][0], board[1][1], board[2][2]]
    yield[board[0][2], board[1][1], board[2][0]]


def emptyspace(board, step):
    '''
    Generate all possible board
    after moving by the step 'step'
    '''
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = step
                yield board
                board[i][j] = 0


class datakeeper(object):
    data = []
    ans = []

    def add(self, board, end):
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

    def fetch(self):
        return array(self.data), array(self.ans)

    def clear(self):
        self.data = []
        self.ans = []

dk = datakeeper()


def randomstep(board, _, __):
    while True:
        i, j = randint(0, 2), randint(0, 2)
        if board[i][j] == 0:
            return i, j


def _complete_check(algorithm, board=None, step=1, ainum=2):
    if board is None:
        board = [[0 for i in range(3)]for j in range(3)]
    log.info('next step: %d by %s. current board:%s' %
             (step, 'AI', printboard(board, None)))
    if 9 >= step >= 6:  # The game may have ended
        end = isend(board, step)
        if end:
            count['AI' if end == ainum else 'Iterator'] += 1
            assert end != ainum, 'end:%s, ainum:%s, but last step was Iterator.' % (end, ainum)
            assert end != 0.5
            if end == ainum:
                log.info('AI wins the game.')
            else:
                log.info('Iterator wins the game.')
            dk.add(deepcopy(board), end % 2)
            return
    elif step == 10:
        count['Draw'] += 1
        log.info('The game ended in draw.')
        dk.add(deepcopy(board), 0.5)
        return
    i, j = algorithm(board, ainum, step)
    assert board[i][j] == 0, 'pos: %d, %d' % (i, j)
    board[i][j] = step
    step += 1
    log.info('next step: %d by %s. current board:%s' %
             (step, 'Iterator', printboard(board, None)))
    if 9 >= step >= 6:  # The game may have ended
        end = isend(board, step)
        if end:
            count['AI' if end == ainum else 'Iterator'] += 1
            board[i][j] = 0
            assert end == ainum, 'end:%s, ainum:%s, but last step was AI.' % (end, ainum)
            if end == ainum:
                log.info('AI wins the game.')
            else:
                log.info('Iterator wins the game.')
            dk.add(deepcopy(board), end % 2)
            return
    elif step == 10:
        count['Draw'] += 1
        board[i][j] = 0
        log.info('The game ended in draw.')
        dk.add(deepcopy(board), 0.5)
        return
    for k, subboard in enumerate(emptyspace(board, step)):
        log.info('enters %dth subboard on step %d' % (k+1, step))
        _complete_check(algorithm, subboard, step+1, ainum)
    board[i][j] = 0


def complete_check(algorithm=algorithm_wiki):
    global count
    dk.clear()
    _complete_check(algorithm, ainum=1)
    print('AI goes first: ' + str(count))
    board = [[0 for i in range(3)] for j in range(3)]
    for subboard in emptyspace(board, 1):
        _complete_check(algorithm, subboard, 2)
    print('Total:' + str(count))
    count = {'AI': 0, 'Iterator': 0, 'Draw': 0}
    return dk

if __name__ == '__main__':
    complete_check()
