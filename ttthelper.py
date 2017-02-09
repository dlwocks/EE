from itertools import product
from ai import perfectalg, label
import random
from numpy import array
from code import interact
from random import randint, shuffle
from tttbase import isend, flatten, deflatten
from copy import deepcopy

def printboard(board, signmap=lambda n: '+' if n==0 else 'O' if n%2 else 'X', hspace='-'):
    '''
    Board printer.
    '''     
    ret = '\n'
    for y, x in product(range(len(board)), range(len(board[0]))):
        ret += str(signmap(board[y][x])) if signmap else str(board[y][x])
        ret += '\n' if x == len(board[0])-1 else hspace
    return ret

def emptyspace_pos(board, step, rnd=False):
    indexlist = list(product(range(3), range(3)))
    if rnd:
        shuffle(indexlist)
    for i, j in indexlist:
        if board[i][j] == 0:
            board[i][j] = step
            yield board, i, j
            board[i][j] = 0


def randomstep(board, _=None, __=None):
    moves = [(i, j) for i in range(3) for j in range(3)]
    random.shuffle(moves)
    while moves:
        i, j = moves.pop()
        if board[i][j] == 0:
            return i, j
    raise RuntimeError('The board passed to randomstep is full')


def gamegen(gamenum, algs=[randomstep] * 2, args=(), piece=True):
    '''
    Generate dataset played with algorithm alg.
    Param:
        gamenum
        algs
        args: additional arguments to put in alg other than board, ainum, step.

    Returned:
        data(2d list): list of boards
        ans(1d list): The final result of corresponding board
    '''
    if not isinstance(algs, list):
        algs = [algs] * 2
    data, ans = [], []
    for i in range(gamenum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = 0
        step = 1
        random.shuffle(algs)
        while not end and step < 10:
            i, j = algs[step % 2](board, (step+1) % 2 + 1, step, *args)
            board[i][j] = step
            if step >= 5:
                end = isend(board, step+1)
                if end is not None:
                    break
            assert step < 9
            step += 1
        if piece:
            data.extend(gen_piece(board))
            ans.extend([end] * step)
        else:
            data.append(flatten(board))
            ans.append(end)
    return data, ans



'''
To train a val network, a board of all zero is not needed because that will not appear in 'next boards'
'''
def gamegen_partial(gamenum, algs=[perfectalg] * 2, args=()):
    '''
    1. Generate a random playout
    2. Pick a step from it, and place the board as is by the step is finished
    3. Play a game starting from that board, played by algs
    4. Record down the board placed in 2 to `data`, the final result in `ans`
    5. Repeat 1-4 as many as gamenum

    Note: Each gamenum in gamegen is equvalent to 7.5 board, when randomstep
    '''
    if not isinstance(algs, list):
        algs = [algs] * 2
    data, ans = [], []
    for i in range(gamenum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = None
        step = 0
        while end is None:
            step += 1
            i, j = randomstep(board)
            board[i][j] = step
            if step >= 5:
                end = isend(board, step+1)
        rndstep = randint(1, step)
        board = gen_piece(board, retmid=rndstep)
        data.append(board)
        board = deflatten(board) # create new copy, no the one appended in data need no copy
        end = isend(board, rndstep+1)
        step = rndstep
        shuffle(algs)
        while end is None:
            step += 1
            i, j = algs[step % 2](board, (step+1) % 2 + 1, step, *args)
            board[i][j] = step
            if step >= 5:
                end = isend(board, step+1)
        ans.append(end)
    return data, ans


def gamegen_pftlabel(boardnum):
    data, ans = [], []
    for i in range(boardnum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = None
        step = 0
        while end is None:
            step += 1
            i, j = randomstep(board)
            board[i][j] = step
            if step >= 5:
                end = isend(board, step+1)
        rndstep = randint(1, step)
        board = gen_piece(board, retmid=rndstep)
        data.append(board)
        ans.append(label(deflatten(board), rndstep+1))
    return data, ans


def adddataset(d1, d2):
    d1[0].extend(d2[0])
    d1[1].extend(d2[1])
    return d1


def gen_piece(board, retmid=None):
    '''
    input:
    [[1,2,3],[4,5,6],[7,8,9]] or [1,2,3,4,5,6,7,8,9]
    output:
    [[1,0,0,0,0,0,0,0,0],
     [1,2,0,0,0,0,0,0,0],
     [1,2,3,0,0,0,0,0,0],
     ...
     [1,2,3,4,5,6,7,8,9]]
    '''
    if isinstance(board[0], list):
        board = flatten(board)
    temp = [0 for i in range(9)]
    ret = []
    for i in range(1, 10):
        try:
            temp[board.index(i)] = i
        except ValueError:
            if i <= 5:
                raise  # The game couldn't have ended!
            break
        if retmid is None:
            ret.append(temp.copy())
        elif i == retmid:
            return temp
    if retmid is None:
        return ret


def extractmove(board):
    '''
    input:
    [[1,2,3],[4,5,6],[7,8,9]] or [1,2,3,4,5,6,7,8,9]
    output:
    [[1,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0],
     ...
     [0,0,0,0,0,0,0,0,1]]
    '''
    if isinstance(board[0], list):
        board = flatten(board)
    ans = []
    for i in range(2, 10):
        try:
            ind = board.index(i)
        except ValueError:
            break
        ans.append([0 if i != ind else 1 for i in range(9)])
    return ans
