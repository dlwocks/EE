from itertools import product
from random import randint, random
from numpy import array


def _row_gen(board):
    '''
    There are 8 ways to end a tic-tac-toe:
    horiziontal 3, vertical 3, diagonal 2.
    This generates every condition.
    '''
    for i in range(3):
        yield [board[i][0], board[i][1], board[i][2]]
        yield [board[0][i], board[1][i], board[2][i]]
    yield [board[0][0], board[1][1], board[2][2]]
    yield [board[0][2], board[1][1], board[2][0]]


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


def isend(board, nx=None):
    """
    Return 1 if the odd player wins the game.
    Return 2 if the even player wins the game.
    Return None elsewise (implicitly)
    """
    for end in map(_isend, _row_gen(board)):
        if end:
            return end
    if nx == 10:
        return 0.5
    elif nx is None:
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return 0
        return 0.5


def emptyspace_pos(board, step):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = step
                yield board, i, j
                board[i][j] = 0


def randomstep(board, _=None, __=None):
    while True:
        i, j = randint(0, 2), randint(0, 2)
        if board[i][j] == 0:
            return i, j


def gamegen(gamenum, alg=randomstep, args=(), noise=0):
    data, ans = [], []
    for i in range(gamenum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = 0
        step = 1
        while not end and step < 10:
            if random() < noise:
                i, j = randomstep(board)
            else:
                i, j = alg(board, (step+1) % 2 + 1, step, *args)
            board[i][j] = step
            if 9 >= step >= 5:
                end = isend(board, step+1)
                if end:
                    break
            step += 1
        data.append(board)
        ans.extend([end if end <= 1 else 0 for _ in range(step)])
    return data, ans


def gen_piece(board):
    '''
    input:
    [1,2,3,4,5,6,7,8,9]
    output:
    [[1,0,0,0,0,0,0,0,0],
     [1,2,0,0,0,0,0,0,0],
     [1,2,3,0,0,0,0,0,0],
     ...
     [1,2,3,4,5,6,7,8,9]]
    '''
    temp = [0 for i in range(9)]
    ret = []
    for i in range(1, 10):
        try:
            temp[board.index(i)] = i
        except ValueError:
            if i <= 5:
                raise  # The game couldn't have ended!
            break
        ret.append(temp[:])
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
    board = list(array(board).flatten())
    ans = []
    for i in range(2, 10):
        try:
            ind = board.index(i)
        except ValueError:
            break
        ans.append([0 if i != ind else 1 for i in range(9)])
    return ans
