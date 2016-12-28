'''Tic-Tac-Toe tester.
Tests the performance of an tic-tac-toe algorithm by complete_check.
It generates every game that may possibly occur when the algorithm is playing
(Given that the algorithm's output only depends on board status rather than, say, random functions)
And counts the number of game that algorithm win, lose, or draw.

Board definition:
The board in this system is a 3*3 array.
0 in the array represent empty space.
Odd number in the array represents 'odd player', who definitely goes first.
Even number in the array represents 'even player', who definitely goes second.
Number n in the board represent nth step played in the game.
'''
import logging as log
from itertools import product, repeat
from numpy import array
from copy import copy, deepcopy
from random import randint

from ai import alg_wiki
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


def _complete_check(algorithm, board=None, step=1, ainum=2):
    if board is None:
        board = [[0 for i in range(3)]for j in range(3)]
    log.info('next step: %d by %s. current board:%s' %
             (step, 'AI', printboard(board, None)))
    if 9 >= step >= 6:  # The game may have ended
        end = isend(board, step)
        if end is not None:
            count['Iterator'] += 1
            assert end != ainum % 2, 'end:%s, ainum:%s, but last step was Iterator.' % (end, ainum)
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
        if end is not None:
            count['AI'] += 1
            board[i][j] = 0
            assert end == ainum % 2, 'end:%s, ainum:%s, but last step was AI.' % (end, ainum)
            assert end != 0.5
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


def complete_check(algorithm=alg_wiki, pt=False):
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


def _board(board):
    return [[0 if not i else 1 if i % 2 else 2 for i in row]for row in board]


def play_with(algorithm, playerfirst=True):
    board = [[0 for i in range(3)]for i in range(3)]
    ainum = 2 if playerfirst else 1
    end = 0
    step = 1
    while not end and step < 10:
        print(printboard(_board(board)))
        if step % 2 == ainum % 2:
            i, j = algorithm(board, ainum, step)
        else:
            i, j = input('please input your step.').split(',')
            i, j = int(i), int(j)
            while board[i][j] != 0:
                i, j = input('the position is occupied. please select another one.').split(',')
                i, j = int(i), int(j)
        board[i][j] = step
        if 9 >= step >= 5:
            end = isend(board, step+1)
            if end is not None:
                if end == 0.5:
                    print('the game ended in draw.')
                elif end % 2 == ainum % 2:
                    print('AI wins the game.')
                else:
                    print("Player wins the game.")
                print(printboard(_board(board)))
                break
        step += 1


if __name__ == '__main__':
    complete_check()
