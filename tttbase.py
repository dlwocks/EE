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

    for end in map(_isend, _row_gen(board)):
        if end is not None:
            return end
    if nx == 10:
        return 0.5  # Next step is 10, thus draw    
    elif nx is not None and nx < 10:
        return None
    elif nx is None:  # nx is unknown, need to check whether game is ended
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return None
        return 0.5


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


def deflatten(board):
    return [board[0:3], board[3:6], board[6:9]]
