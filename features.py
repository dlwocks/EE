def board(board):
    return [0 if not i else 1 if i % 2 else -1 for i in board]


def absboard(board):
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
