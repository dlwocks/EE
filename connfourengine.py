import numpy as np
class Board:
    def __init__(self):
        self._board = np.zeros(42,dtype='int32').reshape(6, 7)
        self._nextmove = 1
        self._isinprocess = True
        self._winner = None

    def __repr__(self):
        board = self._board
        signmap = lambda n: '+' if n == 0 else 'O' if n % 2 else 'X'
        hspace = '-'
        ret = '\n'
        for y in range(len(board)):
            for x in range(len(board[0])):
                ret += signmap(board[y][x])
                ret += '\n' if x == len(board[0])-1 else hspace
        return ret

    def getboard(self):
        return self._board

    def moveat(self, colind):
        if self._winner is not None:
            raise RuntimeError('The game is finished')
        if not 0 <= colind <= 6:
            raise ValueError('invalid index')
        col = self._board[:, colind]
        if col[0]:
            raise ValueError('the column is full')
        for i in range(5, -1, -1):
            if not col[i]:
                col[i] = self._nextmove
                self._nextmove += 1
                break
        if self._nextmove == 43:
            self._winner = 0.5
            return self._winner
        if i <= 2:
            if col[i] % 2 == col[i+1] % 2 == col[i+2] % 2 == col[i+3] % 2:
                self._winner = col[i] % 2
                return self._winner
        row = self._board[i]
        l, r = colind, colind
        while l > 0 and row[l-1] % 2 == col[i] % 2 and row[l-1]:
            l -= 1
        while r < 6 and row[r+1] % 2 == col[i] % 2 and row[r+1]:
            r += 1
        if r - l + 1 >= 4:
            self._winner = col[i] % 2
            return self._winner
        # diagnoal /
        l, r, t, b = colind, colind, i, i
        while l > 0 and b < 5 and self._board[b+1][l-1] % 2 == col[i] % 2 and self._board[b+1][l-1]:
            b += 1
            l -= 1
        while r < 6 and t > 0 and self._board[t-1][r+1] % 2 == col[i] % 2 and self._board[t-1][r+1]:
            t -= 1
            r += 1
        if b - t + 1 >= 4:
            self._winner = col[i] % 2
            return self._winner
        # diagonal \
        l, r, t, b = colind, colind, i, i
        while l > 0 and t > 0 and self._board[t-1][l-1] % 2 == col[i] % 2 and self._board[t-1][l-1]:
            t -= 1
            l -= 1
        while r < 6 and b < 5 and self._board[b+1][r+1] % 2 == col[i] % 2 and self._board[b+1][r+1]:
            b += 1
            r += 1
        if r - l + 1 >= 4:
            self._winner = col[i] % 2
            return self._winner

    def winner(self):
        return self._winner

    def nextplayer(self):
        return self._nextmove % 2
