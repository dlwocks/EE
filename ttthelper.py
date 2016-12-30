from itertools import product
import random
from numpy import array


INT = "__import__('code').interact(local=locals())"


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
    if not all(nums):
        return None
    if nums[0] % 2 == nums[1] % 2 == nums[2] % 2:
        return 1 if nums[0] % 2 else 0
    return None


def isend(board, nx=None):
    """
    Determine Whether a game has ended in a specific ttt board.
    Make sure that the board is produced in ttt play following rule. 
    i.e. if the board has both odd and even player winning, it will only return the first detected winning.
    Param:
        board: 2d.
        nx: number for next step. Optional - speed up progress.
    Returned:
        Return 1 if the odd player wins the game.
        Return 0 if the even player wins the game.
        Return 0.5 if the game ended in draw.
        Return None if the game hasn't ended.
    """
    if nx == 10:
        return 0.5  # Next step is 10, thus draw
    for end in map(_isend, _row_gen(board)):
        if end is not None:
            return end
    if nx is not None:
        return None  # nx should be smaller than 10, but no row ends the game, so the game continues
    else:  # nx is unknown, need to check whether game is ended
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return None
        return 0.5


def emptyspace_pos(board, step):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = step
                yield board, i, j
                board[i][j] = 0


def randomstep(board, _=None, __=None):
    moves = [(i,j) for i in range(3) for j in range(3)]
    random.shuffle(moves)
    while moves:
        i, j = moves.pop()
        if board[i][j] == 0:
            return i, j
    raise RuntimeError('The board passed to randomstep is full')


def flatten(board):
    '''
    input: [[1,2,3],[4,5,6],[7,8,9]]
    output: [1,2,3,4,5,6,7,8,9]
    '''
    if isinstance(board[0], int):
        if len(board) == 9:
            return board  # Already flattened
        else:
            raise ValueError('are you sure this is a board?')
    return board[0] + board[1] + board[2]



def gamegen(gamenum, algs=[randomstep] * 2, args=()):
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
    if not isinstance(algs, list) or len(algs) != 2:
        raise ValueError('param alg is not correct')
    data, ans = [], []
    for i in range(gamenum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = 0
        step = 1
        random.shuffle(algs)
        while not end and step < 10:
            i, j = algs[step % 2](board, (step+1) % 2 + 1, step, *args)
            board[i][j] = step
            if 9 >= step >= 5:
                end = isend(board, step+1)
                if end is not None:
                    break
            step += 1
        data.extend(gen_piece(board))
        ans.extend([end] * step)
    return data, ans


def gen_piece(board):
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
        ret.append(temp.copy())
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
