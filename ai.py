import logging as log
from itertools import product


def _permutation_3(row):
    yield row[0], row[1], row[2]
    yield row[1], row[2], row[0]
    yield row[0], row[2], row[1]


def _two_in_a_row(board, ainum):
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


def _fork(board, ainum, turn):
    tempboard = [[-100 if board[j][i] else 0 for i in range(3)]for j in range(3)]
    for row in _row_gen_pos():
        for i, j, k in _permutation_3(row):
            if board[i[0]][i[1]] == board[j[0]][j[1]] == 0 and board[k[0]][k[1]] != 0 and board[k[0]][k[1]] % 2 == ainum % 2:
                tempboard[i[0]][i[1]] += 1
                tempboard[j[0]][j[1]] += 1
                continue
    return [pos for pos in product(range(3), range(3)) if tempboard[pos[0]][pos[1]] >=2]


def _fork_opponent(board, ainum, turn):
    oppofork = _fork(board, (ainum % 2) + 1, turn)
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


def _row_gen_pos():
    for i in range(3):
        yield [(i, 0), (i, 1), (i, 2)]
        yield [(0, i), (1, i), (2, i)]
    yield [(0, 0), (1, 1), (2, 2)]
    yield [(0, 2), (1, 1), (2, 0)]


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
    pos = _two_in_a_row(board, ainum)
    if pos:
        log.info('returned in _two_in_a_row')
        return pos
    pos = _fork(board, ainum, turn)
    if pos:
        log.info('returned in own fork')
        return pos[0]
    pos = _fork_opponent(board, ainum, turn)
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
    assert False
