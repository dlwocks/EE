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
from ttthelper import isend, printboard

global count

log.basicConfig(filename='ttttester.log',
                filemode='w',
                level=log.INFO,
                format='%(asctime)s %(message)s')
RETURNBEFORE = False
count = {'AI': 0, 'Iterator': 0, 'Draw': 0}


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
            count['Iterator'] += 1
            assert end != ainum, 'end:%s, ainum:%s, but last step was Iterator.' % (end, ainum)
            assert end != 0.5
            log.info('Iterator wins the game.')
            return
    elif step == 10:
        count['Draw'] += 1
        log.info('The game ended in draw.')
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
            count['AI'] += 1
            board[i][j] = 0
            assert end == ainum, 'end:%s, ainum:%s, but last step was AI.' % (end, ainum)
            log.info('AI wins the game.')
            return
    elif step == 10:
        count['Draw'] += 1
        board[i][j] = 0
        log.info('The game ended in draw.')
        return
    for k, subboard in enumerate(emptyspace(board, step)):
        log.info('enters %dth subboard on step %d' % (k+1, step))
        _complete_check(algorithm, subboard, step+1, ainum)
    board[i][j] = 0


def complete_check(algorithm=algorithm_wiki, pt=True):
    global count
    count = {'AI': 0, 'Iterator': 0, 'Draw': 0}
    _complete_check(algorithm, ainum=1)
    if pt:
        print('AI goes first: ' + str(count))
    board = [[0 for i in range(3)] for j in range(3)]
    for subboard in emptyspace(board, 1):
        _complete_check(algorithm, subboard, 2)
    if pt:
        print('Total:%s' % str(count))
        print('Algorithm Evaluation Point:%0.2f' % ((count['AI'] * 2 + count['Draw'])/sum(count.values())))
    return (count['AI'] * 2 + count['Draw'])/sum(count.values())

if __name__ == '__main__':
    complete_check()
