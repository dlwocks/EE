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


def _permutation_3(row):
    yield row[0], row[1], row[2]
    yield row[1], row[2], row[0]
    yield row[0], row[2], row[1]


def two_in_a_row(board, ainum):
    '''
    Return one of positions that completes a two-in-a-row.
    (That is, a position either wins or blocks)
    Returns the one make ainum win if both player has two-in-a-row.
    '''
    temp = None
    for row in _row_gen_pos():
        for i, j, k in _permutation_3(row):
            if board[i[0]][i[1]] % 2 == board[j[0]][j[1]] % 2 and board[i[0]][i[1]] != 0 and board[j[0]][j[1]] != 0 and board[k[0]][k[1]] == 0:
                if board[i[0]][i[1]] % 2 == ainum % 2:
                    return k
                else:
                    temp = k
    log.info(repr(temp))
    return temp


def fork(board, ainum, turn):
    tempboard = [[-100 if board[j][i] else 0 for i in range(3)]for j in range(3)]
    for row in _row_gen_pos():
        for i, j, k in _permutation_3(row):
            if board[i[0]][i[1]] == board[j[0]][j[1]] == 0 and board[k[0]][k[1]] != 0 and board[k[0]][k[1]] % 2 == ainum % 2:
                tempboard[i[0]][i[1]] += 1
                tempboard[j[0]][j[1]] += 1
                continue
    return [pos for pos in product(range(3), range(3)) if tempboard[pos[0]][pos[1]] >=2]


def fork_opponent(board, ainum, turn):
    oppofork = fork(board, (ainum % 2) + 1, turn)
    if not oppofork:
        return None
    create_tiar = set()
    for row in _row_gen_pos():
        for i, j, k in _permutation_3(row):
            if board[i[0]][i[1]] == board[j[0]][j[1]] == 0 != board[k[0]][k[1]] and board[k[0]][k[1]] % 2 == ainum % 2:
                create_tiar.add((i, j))
    for posone, postwo in create_tiar:
        if posone not in oppofork:
            return postwo
        if postwo not in oppofork:
            return posone
    return oppofork[0]


def algorithm_wiki(board, ainum, turn):
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
    pos = two_in_a_row(board, ainum)
    if pos:
        log.info('returned in two_in_a_row')
        return pos
    pos = fork(board, ainum, turn)
    if pos:
        log.info('returned in own fork')
        return pos[0]
    pos = fork_opponent(board, ainum, turn)
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
    for i in [0, 2]:
        for j in [0, 2]:
            if board[i][j] == 0:
                return i, j
    for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        if board[i][j] == 0:
            return i, j
    assert RETURNBEFORE


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


def isend(board):
    """
    Return 1 if the odd player wins the game.
    Return 2 if the even player wins the game.
    Return None elsewise (implicitly)
    """
    for end in map(_isend, _row_gen(board)):
        if end:
            return end


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


def _row_gen_pos():
    for i in range(3):
        yield [(i, 0), (i, 1), (i, 2)]
        yield [(0, i), (1, i), (2, i)]
    yield [(0, 0), (1, 1), (2, 2)]
    yield [(0, 2), (1, 1), (2, 0)]


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


def _exhaustive_check(algorithm, board=None, step=1, ainum=2):
    if board is None:
        board = [[0 for i in range(3)]for j in range(3)]
    log.info('next step: %d by %s. current board:%s' %
             (step, 'AI', printboard(board, None)))
    end = isend(board)
    if end:
        count['AI' if end == ainum else 'Iterator'] += 1
        assert end != ainum, 'end:%s, ainum:%s, but last step was Iterator.' % (end, ainum)
        if end == ainum:
            log.info('AI wins the game.')
        else:
            log.info('Iterator wins the game.')
        dk.add(deepcopy(board), 1 if end == 1 else 0)
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
    end = isend(board)
    if end:
        count['AI' if end == ainum else 'Iterator'] += 1
        board[i][j] = 0
        assert end == ainum, 'end:%s, ainum:%s, but last step was AI.' % (end, ainum)
        if end == ainum:
            log.info('AI wins the game.')
        else:
            log.info('Iterator wins the game.')
        dk.add(deepcopy(board), 1 if end == 1 else 0)
        return
    elif step == 10:
        count['Draw'] += 1
        board[i][j] = 0
        log.info('The game ended in draw.')
        return
    for k, subboard in enumerate(emptyspace(board, step)):
        log.info('enters %dth subboard on step %d' % (k+1, step))
        _exhaustive_check(algorithm, subboard, step+1, ainum)
    board[i][j] = 0


def exhaustive_check(algorithm=algorithm_wiki):
    global count
    dk.clear()
    _exhaustive_check(algorithm, ainum=1)
    print('AI goes first: ' + str(count))
    board = [[0 for i in range(3)] for j in range(3)]
    for subboard in emptyspace(board, 1):
        _exhaustive_check(algorithm, subboard, 2)
    print('Total:' + str(count))
    count = {'AI': 0, 'Iterator': 0, 'Draw': 0}
    return dk

if __name__ == '__main__':
    exhaustive_check()
