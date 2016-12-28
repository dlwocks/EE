'''
A myth to resolve:
Why logreg_like ann is so slow compared to logreg??


Researches can be done:
1.Incresing hidden unit           val_hidden = [...]
2.feeding better features         gamegen(num = ...)
3.feeding more features           gamegen(alg = [...])
4.feeding better data             ann_ai(features = ...)
5.feeding more data               features.py ...
6.use/mix poilcy&value network    pol_hidden = []; if self.USE_VAL and self.USE_POL: ...
'''
def script_lr(num=100):
    # Test a logreg_ai
    from logreg_ai import logreg_ai
    import ttttester
    import ttthelper
    lrai = logreg_ai()
    lrai.startlearn(game=num, pt=False, graph=False)
    return ttttester.complete_check(lrai.getstep)

lr = [script_lr() for i in range(10)]

def script_lrann(num=100):  # r2
    # Script 1: Create a logreg_like ann with random dataset of game number num
    from ann_ai import ann_ai
    import ttttester
    import ttthelper
    ai = ann_ai(val_hidden=[])
    ai.train(dataset=ttthelper.gamegen(gamenum=num), pt=False)
    return ttttester.complete_check(ai.getstep)

lrann = [script_lrann(num=1000) for i in range(100)]


def script_lrann2(num=100):
    # Train logreg_like ann in the logreg way
    from ann_ai import ann_ai
    import ttttester
    import ttthelper
    dataset = [[], []]
    ai = ann_ai(val_hidden=[], cython=True)
    for i in range(num):
        subdataset = ttthelper.gamegen(1, algs=[ai.getstep, ttthelper.randomstep])
        dataset[0] += subdataset[0]
        dataset[1] += subdataset[1]
        ai.train(dataset=dataset, pt=False)
    return ttttester.complete_check(ai.getstep)


# Useful to do some repetitive job until some time!
from datetime import datetime
from itertools import count
UNTIL = datetime(2016, 12, 27, 7, 5, 0)
lrann = []
lrann2 = []
for i in count():
    ini = datetime.now()
    # Put your job over here
    lrann.append(script_lrann(num=1000))
    lrann2.append(script_lrann2(num=1000))
    #
    fin = datetime.now()
    if (UNTIL - fin) < (fin - ini) * 1.2:
        break


def alg_timer(alg):
    # Times the average times required to take a step
    from timeit import timeit
    import ttthelper
    time = 0
    GAME = 10
    NUM = 100
    timedcount = 0
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for _ in range(GAME):
        step = 1
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        while ttthelper.isend(board) is None:
            time += timeit(lambda: alg(board, 1 if step % 2 else 2, step), number=NUM)
            timedcount += 1
            i, j = alg(board, 1 if step % 2 else 2, step)
            board[i][j] = step
    return time / timedcount / NUM






def script_ann():
    from ann_ai import ann_ai
    import ttttester
    import ttthelper