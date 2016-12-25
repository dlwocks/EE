'''
Specifies features.

All function here transforms a flattened tic-tac-toe board(as defined in ttttester.py) into another board using its own rule.
'''
def board(board):
    '''Linear: Odds -> 1, Evens -> -1, None - > 0'''
    return [0 if not i else 1 if i % 2 else -1 for i in board]


def extboard(board):
    '''Half-Linear: len=18; First half: Odds -> 1, Else -> 0. Second half: Evens -> 1, Else -> 0'''
    return [1 if i % 2 else 0 for i in board] + [1 if not i % 2 else 0 for i in board]


def absboard(board):
    '''Linear: None -> 0, Else -> 1'''
    return [0 if not i else 1 for i in board]


def nboard(board):
    m = max(board)
    return [0 if not i or i % 2 == m % 2 else 1 if i % 2 else -1 for i in board]


def lboard(board):
    m = max(board)
    return [0 if not i or i % 2 != m % 2 else 1 if i % 2 else -1 for i in board]


def oboard(board):
    return [0 if not i else i if i % 2 else -i for i in board]


def orboard(board):
    m = max(board)
    return [0 if not i else (m - i + 1) if i % 2 else -(m - i + 1) for i in board]


def nextplayer(board):
    return [1 if max(board) % 2 else -1]

ctsur_map = {0: [1, 3, 4],
             1: [0, 2, 3, 4, 5],
             2: [1, 4, 5],
             3: [0, 1, 4, 6, 7],
             4: [0, 1, 2, 3, 5, 6, 7, 8],
             5: [1, 2, 4, 7, 8],
             6: [3, 4, 7],
             7: [3, 4, 5, 6, 8],
             8: [4, 5, 7]}


def ctsur(board):  # with board: 1.06
    ret = [0 for i in range(18)]
    for i in range(9):
        for p in ctsur_map[i]:
            if board[p] != 0 and board[p] % 2 == 0:
                ret[i] += 1
            elif board[p] % 2 == 1:
                ret[i+9] += 1
    return ret
