from code import interact
import ttthelper
import ann_ai
import ttttester
from ai import perfectalg
import matplotlib.pyplot as plt
import numpy as np
import itertools
import statistics as stat
from datetime import datetime, timedelta
from learningfunc import ann
import pickle
import os
from random import shuffle, randint
import tttbase
from ttttester import completecheck, randomcheck
from ttthelper import gamegen_partial, randomstep, gamegen, adddataset
from itertools import count

def mean(ls, ignoremax=0, ignoremin=0, formin=None, formax=None):
    if len(ls) - ignoremax - ignoremin < 1:
        raise RuntimeError
    if not (ignoremax or ignoremin or formin or formax):
        return stat.mean([i for i in ls if not np.isnan(i)])
    ls.sort()
    if ignoremax or ignoremin:
        return stat.mean([i for i in ls[ignoremin: -ignoremax]])
    if formin:
        return stat.mean(ls[:formin])
    elif formax:
        return stat.mean(ls[-formax:])

def stdev(ls):
    return stat.stdev(ls)


class timer(object):
    def __init__(self):
        self._veryini = datetime.now()
        print('Started at', self._veryini)

    def __call__(self):
        return str(datetime.now() - self._veryini)

    def __enter__(self):
        if not hasattr(self, '_acctime'):
            self._acctime = timedelta()
        if not hasattr(self, '_num'):
            self._num = 0
        self._ini = datetime.now()

    def __exit__(self, _, __, ___):
        self._time = datetime.now() - self._ini
        self._acctime += self._time
        self._num += 1
        print('Single: %s   Acc: %s   Num: %s' % (self.time, self.acc, self.num))

    @property
    def time(self):
        return str(self._time)

    @property
    def acc(self):
        return str(self._acctime)

    @property
    def num(self):
        return self._num


'''
It is shown that a large number of random gamegen(larger than 1000) is able to
train a simple 9-hidden-unit-board-featured-ai close to perfect. (AEP larger than 90)
Thus, this script **tries to find maximum AEP that could be get from the training with same config.**
The highest AEP observed is: 92.77 (6000 game, 9 hidden)
The perfect one has AEP of: 91.89
'''
def r1(gamenum=2000, hidden=[9], feature=['board']):  # thesre are the specific config.
    t = timer()
    if isinstance(hidden, int):
        hidden = [hidden]
    allptrec = []
    while True:
        try:
            with t:
                dataset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                ai.train(dataset, pt=False)
                pt = ttttester.complete_check(ai.getstep)[0]
            allptrec.append(pt)
        except KeyboardInterrupt:
            break
    if len(allptrec) >= 10:
        plt.hist(allptrec)
        plt.show()
    interact(local=locals())


'''
Seems like there is still a limit AEP that can be trained with this config. Then we change it.
To find the best config, first we **find the effect of number of training input on performance.**

6000-7000 games seems to show good average AEP
'''
def r2(ini=1000, step=1000, num=10):  # inis for gamenum
    t = timer()
    gamenumlist = [ini + step * i for i in range(num)]
    ptrec = [[] for i in range(num)]
    while True:
        try:
            with t:
                for i, gamenum in enumerate(gamenumlist):
                    dataset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                    ai = ann_ai.ann_ai(val_hidden=[9], pol_hidden=None, feature=['board'])
                    ai.train(dataset, pt=False)
                    pt = ttttester.complete_check(ai.getstep)
                    ptrec[i].append(pt)
            print('One iteration done in %s' % t())
        except KeyboardInterrupt:
            break
    ptrecmean = [mean(thisrec) for thisrec in ptrec]
    plt.plot(gamenumlist, ptrecmean)
    plt.show()
    interact(local=locals())

'''
Also, we can **find the effect of number of hidden layer on performance**.

Good performance in about 8-10 hidden units.
Why adding hidden unit above the value helps negatively??
    -Overfitting..? Need to investigate this upon minres.fun
'''
def r3(ini=1, step=1, num=20):  # inis for hidden layer
    t = timer()
    if ini == 0:
        hiddenslist = [[]] + [[ini + step * i] for i in range(1, num)]
    elif ini > 0:
        hiddenslist = [[ini + step * i] for i in range(num)]
    gamenum = 1000
    ptrec = [[] for _ in range(num)]
    while True:
        try:
            with t:
                dataset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                for j, hidden in enumerate(hiddenslist):
                    ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                    ai.train(dataset, pt=False)
                    pt = ttttester.complete_check(ai.getstep)[0]
                    ptrec[j].append(pt)
            print('Done in %s' % t())
        except KeyboardInterrupt:
            break
    ptrecmean = [mean(thisrec) for thisrec in ptrec]
    plt.plot([ini + step * i for i in range(num)], ptrec)
    plt.title()
    plt.show()
    interact(local=locals())


'''
I haven't considered to measure the error of ann directly. AEP is not an accurate measure on the accuracy of the ann
Following script will directly measure the error of ann to **choose the number of hidden layer**.
It is similar to r3, but use error of ann on **validation set** as measure.
Value Network specific.

10-16 hidden unit seems to be optimal.
'''
def r4():
    ini = 1
    step = 1
    num = 20
    fileexist = os.path.exists('r4.dump')
    if fileexist:
        print('dumped file found')
        with open('r4.dump', 'rb') as o:
            trainerrs, validateerrs, setup = pickle.load(o)
    if not fileexist:
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]
    if fileexist and setup != (ini, step, num):
        input('Setup doesn\'t match. You sure continue?')
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainnum = 1000
    while True:
        try:
            with t:
                trainset = ttthelper.gamegen(trainnum, algs=[ttthelper.randomstep] * 2)
                validateset = ttthelper.gamegen(trainnum//4, algs=[ttthelper.randomstep] * 2)
                for i, hidden in enumerate(hiddenslist):
                    ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                    minres = ai.train(trainset, pt=False)
                    trainerrs[i].append(minres.fun)
                    cost = ai.getcost(validateset)
                    validateerrs[i].append(cost)
        except KeyboardInterrupt:
            break
    trainerrmean = [min(thiserr) for thiserr in trainerrs]
    validateerrmean = [min(thiserr) for thiserr in validateerrs]
    p1 = plt.plot([ini + step * i for i in range(num)], trainerrmean)
    p2 = plt.plot([ini + step * i for i in range(num)], validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    plt.show()
    with open('r4.dump', 'wb') as o:
        setup = (ini, step, num)
        pickle.dump((trainerrs, validateerrs, setup), o)
    interact(local=locals())

def r4_l(seclayer=10):
    ini = 10
    step = 2
    num = 5
    trainerrs = [[] for i in range(num)]
    validateerrs = [[] for i in range(num)]
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainnum = 1000
    while True:
        try:
            with t:
                trainset = ttthelper.gamegen(trainnum, algs=[ttthelper.randomstep] * 2)
                validateset = ttthelper.gamegen(trainnum//4, algs=[ttthelper.randomstep] * 2)
                for i, hidden in enumerate(hiddenslist):
                    ai = ann_ai.ann_ai(val_hidden=[hidden, seclayer], pol_hidden=None, feature=['board'])
                    minres = ai.train(trainset, pt=False)
                    trainerrs[i].append(minres.fun)
                    cost = ai.getcost(validateset)
                    validateerrs[i].append(cost)
        except KeyboardInterrupt:
            break
    trainerrmean = [min(thiserr) for thiserr in trainerrs]
    validateerrmean = [min(thiserr) for thiserr in validateerrs]
    p1 = plt.plot([ini + step * i for i in range(num)], trainerrmean)
    p2 = plt.plot([ini + step * i for i in range(num)], validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    plt.show()
    interact(local=locals())

def r4_2d():
    ini = 4
    step = 2
    num = 6
    trainerrs = [[[] for i in range(num)] for i in range(num)]
    valierrs = [[[] for i in range(num)] for i in range(num)]
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainnum = 1000
    while True:
        try:
            with t:
                trainset = ttthelper.gamegen(trainnum, algs=[ttthelper.randomstep] * 2)
                validateset = ttthelper.gamegen(trainnum//4, algs=[ttthelper.randomstep] * 2)
                for i, hid1 in enumerate(hiddenslist):
                    for j, hid2 in enumerate(hiddenslist):
                        ai = ann_ai.ann_ai(val_hidden=[hid1, hid2], pol_hidden=None, feature=['board'])
                        minres = ai.train(trainset, pt=False)
                        trainerrs[i][j].append(minres.fun)
                        cost = ai.getcost(validateset)
                        valierrs[i][j].append(cost)
                    print('1st hid %s done in %s' % (str(hid1), t()))
        except KeyboardInterrupt:
            break
    interact(local=locals())
    valierrmean = [[min(thiserrs) for thiserrs in hiddenerrs] for hiddenerrs in valierrs]
    temp = []
    for i, hiddenerrmean in enumerate(valierrmean):
        p = plt.plot([ini + step * i for i in range(num)], valierrmean[i])
        temp.append(p[0])
    plt.legend(temp, ['1st hid: %s' % str(h) for h in hiddenslist])
    plt.show()
    interact(local=locals())


'''
The optimal number of hidden layer will be determined above.
Now this script will measure the error of ann to **choose the number of training set**.
It is similar to r2, but use error of ann on **validation set** as measure.
Value Network specific.
To Think: Should validateset be outside of gamenumlist loop?

Over 5000-6000 training set seems to produce minimal improvement.
The game number may be improved by using training set from more educated alg
'''
def r5(ini=1000, step=1000, num=10):
    fileexist = os.path.exists('r5.dump')
    if fileexist:
        print('dumped file found')
        with open('r5.dump', 'rb') as o:
            trainerrs, validateerrs, setup = pickle.load(o)
    if not fileexist:
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]
    if fileexist and setup != (ini, step, num):
        input('Setup doesn\'t match. You sure continue?')
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]  
    OPTIMAL_LAYERNUM = 8
    t = timer()
    gamenumlist = [ini + step * i for i in range(num)]
    while True:
        try:
            with t:
                validateset = ttthelper.gamegen(ini//2, algs=[ttthelper.randomstep] * 2)
                for i, gamenum in enumerate(gamenumlist):
                    trainset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                    ai = ann_ai.ann_ai(val_hidden=[OPTIMAL_LAYERNUM], pol_hidden=None, feature=['board'])
                    minres = ai.train(trainset, pt=False)
                    trainerrs[i].append(minres.fun)  # 2 as from gamegen(ini//`2`)
                    validateerrs[i].append(ai.getcost(validateset))
            print('Single: %s   Acc: %s   Num: %s' % (t.time, t.acc, t.num))
        except KeyboardInterrupt:
            break
    trainerrmean = [mean(thiserr) for thiserr in trainerrs]
    validateerrmean = [mean(thiserr) for thiserr in validateerrs]
    p1 = plt.plot([ini + step * i for i in range(num)], trainerrmean)
    p2 = plt.plot([ini + step * i for i in range(num)], validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    plt.show()
    plt.plot([ini + step * i for i in range(num)], np.array(validateerrmean) - np.array(trainerrmean))
    plt.show()
    with open('r5.dump', 'wb') as o:
        setup = (ini, step, num)
        pickle.dump((trainerrs, validateerrs, setup), o)
    interact(local=locals())


'''
Regularization!

Result saying, regularization may be useless in this
'''
def r6(ini=0, step=0.02, num=30):
    fileexist = os.path.exists('r6.dump')
    if fileexist:
        print('dumped file found')
        with open('r6.dump', 'rb') as o:
            trainerrs, validateerrs, setup = pickle.load(o)
    if not fileexist:
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]
    if fileexist and setup != (ini, step, num):
        input('Setup doesn\'t match. You sure continue?')
        trainerrs = [[] for i in range(num)]
        validateerrs = [[] for i in range(num)]  
    GAMENUM = 500
    OPTLAYER = 9
    t = timer()
    reglist = [ini + step * i for i in range(num)]
    trainerrs = [[] for i in range(num)]
    validateerrs = [[] for i in range(num)]
    while True:
        try:
            with t:
                trainset = ttthelper.gamegen(GAMENUM, algs=[ttthelper.randomstep] * 2)
                validateset = ttthelper.gamegen(GAMENUM // 2, algs=[ttthelper.randomstep] * 2)
                for i, reg in enumerate(reglist):
                    ai = ann_ai.ann_ai(val_hidden=[OPTLAYER], pol_hidden=None, feature=['board'], reg=reg)
                    minres = ai.train(trainset, pt=False)
                    trainerrs[i].append(minres.fun)
                    validateerrs[i].append(ai.getcost(validateset))
            print('Single: %s   Acc: %s   Num: %s' % (t.time, t.acc, t.num))
        except KeyboardInterrupt:
            break
    with open('r6.dump', 'wb') as o:
        setup = (ini, step, num)
        pickle.dump((trainerrs, validateerrs, setup), o)
    interact(local=locals())

'''
Test for different way of producing dataset
'''
def r7():
    boardnum = 10000
    from ann_ai import ann_ai
    dataset1 = gamegen_partial(boardnum, algs=randomstep)
    dataset2 = gamegen_partial(boardnum, algs=perfectalg)
    tempdataset = gamegen(100)
    tempai = ann_ai(val_hidden=11)
    tempai.train(tempdataset)
    dataset3 = gamegen_partial(boardnum, algs=tempai.getstep)
    print('randomstep:')
    print(completecheck(randomstep), randomcheck(randomstep))
    print('slightly educated ai:')
    print(completecheck(tempai.getstep), randomcheck(tempai.getstep))
    dataset4 = [], []
    for i in range(100):
        tempdataset = gamegen(100)
        tempai = ann_ai(val_hidden=11)
        tempai.train(tempdataset)
        dataset4 = adddataset(dataset4, gamegen_partial(boardnum//100, algs=tempai.getstep))
    airnd = ann_ai(val_hidden=11)
    aipft = ann_ai(val_hidden=11)
    aised = ann_ai(val_hidden=11)
    aiseds = ann_ai(val_hidden=11)
    airnd.train(dataset1)
    print('ai trained with random dataset:')
    print(completecheck(airnd.getstep), randomcheck(airnd.getstep))
    aipft.train(dataset2)
    print('ai trained with perfect dataset:')
    print(completecheck(aipft.getstep), randomcheck(aipft.getstep))
    aised.train(dataset3)
    print('ai trained with slightly educated dataset:')
    print(completecheck(aised.getstep), randomcheck(aised.getstep))
    aiseds.train(dataset4)
    print('ai trained with multiple slightly educated dataset:')
    print(completecheck(aiseds.getstep), randomcheck(aiseds.getstep))
    print('perfect ai:')
    print(completecheck(perfectalg), randomcheck(perfectalg))
    print('\a')
    interact(local=locals())


def r7_2():
    ini = 10
    step = 1
    num = 21
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainnum = 10000
    alglist = [randomstep, perfectalg]
    trainerrs = [[[] for i in range(num)] for i in range(len(alglist))]
    valierrs = [[[] for i in range(num)] for i in range(len(alglist))]
    while True:
        try:
            with t:
                for a, alg in enumerate(alglist):
                    trainset = gamegen_partial(trainnum, algs=alg)
                    valiset = gamegen_partial(trainnum//4, algs=alg)
                    for i, hidden in enumerate(hiddenslist):
                        ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                        minres = ai.train(trainset, pt=False)
                        trainerrs[a][i].append(minres.fun)
                        cost = ai.getcost(valiset)
                        valierrs[a][i].append(cost)
        except KeyboardInterrupt:
            break
    trainerrmean = [[mean(thiserr) for thiserr in algerrs] for algerrs in trainerrs]
    valierrmean = [[mean(thiserr) for thiserr in algerrs] for algerrs in valierrs]
    temp = []
    for algerrs in valierrmean:
        p = plt.plot([ini + step * i for i in range(num)], algerrs)
        temp.append(p[0])
    plt.legend(temp, [alg.__name__ for alg in alglist])
    plt.show()
    interact(local=locals())

def r7_3():
    # Dumped: (ailist, trainerrs, valierrs)
    '''
    If validation cost decrease but performance worsen, then there should be a problem.
    The only difference is hidden layer number; validation set no change, so cost can be compared.
    '''
    trainnum = 10000
    trainset = gamegen_partial(trainnum, algs=perfectalg)
    valiset = gamegen_partial(trainnum//4, algs=perfectalg)
    ini = 10
    step = 4
    num = 9
    hiddenslist = [ini + step * i for i in range(num)]
    ailist = [[] for i in range(num)]
    trainerrs = [[] for i in range(num)]
    valierrs = [[] for i in range(num)]
    checkreslist = [[] for i in range(num)]
    winrate = [[] for i in range(num)]
    t = timer()
    while True:
        try:
            with t:
                for i, hidden in enumerate(hiddenslist):
                    ai = ann_ai.ann_ai(val_hidden=hidden)
                    minres = ai.train(trainset)
                    trainerrs[i].append(minres.fun)
                    cost = ai.getcost(valiset)
                    valierrs[i].append(cost)
                    checkres = randomcheck(ai)
                    checkreslist[i].append(checkres)
                    winrate[i].append(checkres[0]/10000)
                    ailist[i].append(ai)
        except KeyboardInterrupt:
            break
    trainmean = [mean(t) for t in trainerrs]
    valimean = [mean(v) for v in valierrs]
    winratemean = [mean(r) for r in winrate]
    p1, p2, p3 = plt.plot(hiddenslist, trainmean), plt.plot(hiddenslist, valimean), plt.plot(hiddenslist, winratemean)
    plt.legend((p1[0], p2[0], p3[0]), ('train', 'validate', 'win rate'))
    plt.show()
    interact(local=locals())






'''
Test for different features!
With different features, comparing error is meaningless, isn't it..?
    Or maybe no, error just describes how well the model can predict final result
    So as long as dataset is produced in same way, the cost would be comparable.
One point: if feature different, suitable hidden layer should be different!
'''
def r8(*features):
    ini = 6
    step = 2
    num = 5
    trainerrs = [[[] for i in range(num)] for i in range(len(features))]
    validateerrs = [[[] for i in range(num)] for i in range(len(features))]
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainnum = 10000
    while True:
        try:
            with t:
                trainset = gamegen_policy(trainnum)
                validateset = gamegen_policy(trainnum//4)
                for i, hidden in enumerate(hiddenslist):
                    for f, feature in enumerate(features):
                        ai = ann_ai.ann_ai(pol_hidden=hidden, feature=feature)
                        minres = ai.train(trainset)
                        trainerrs[f][i].append(minres.fun)
                        cost = ai.getcost(validateset)
                        validateerrs[f][i].append(cost)
                    print('hidden layer %d done in %s' % (hidden, t()))
        except KeyboardInterrupt:
            break
    trainp, valip = [], []
    for train, validate, feature in zip(trainerrs, validateerrs, features):
        #trainerrmean = [mean(thiserr) for thiserr in train]
        validateerrmean = [mean(thiserr) for thiserr in validate]
        #trainp.append(plt.plot([ini + step * i for i in range(num)], trainerrmean)[0])
        valip.append(plt.plot([ini + step * i for i in range(num)], validateerrmean)[0])

    plt.legend(valip, ['%s: Validate' % str(f) for f in features])
    plt.show()
    interact(local=locals())

'''
Policy Network! I think this is the last r needed for ttt

To create dataset for policy network:
1. Create one random situation
2. Check perfectalg's move at the chosen situation
3. Append situation as data, append move as ans
4. Repeat 1-3
'''
def gamegen_policy(boardnum):
    data = []
    ans = []
    for i in range(boardnum):
        board = [[0 for i in range(3)]for i in range(3)]
        end = None
        step = 0
        while end is None:
            step += 1
            i, j = ttthelper.randomstep(board)
            board[i][j] = step
            if step >= 5:
                end = tttbase.isend(board, step+1)
        rndstep = randint(1, 8 if step >= 9 else step)
        board = ttthelper.gen_piece(board, retmid=rndstep)
        data.append(board.copy())
        i, j = perfectalg(tttbase.deflatten(board), 1 if rndstep+1 % 2 else 2, rndstep+1, rndfrombest=True)
        mv = [1 if ind == 3 * i + j else 0 for ind in range(9)]
        ans.append(mv)
    return data, ans

def r9_single(hidden=40):
    dataset = gamegen_policy(30000)
    ai = ann_ai.ann_ai(pol_hidden=hidden, feature=['board', 'nextplayer'])
    minres = ai.train(dataset)
    print('training completed')
    print(completecheck(ai.getstep))
    print(randomcheck(ai.getstep))
    interact(local=locals())

def r9(number=3):
    t = timer()
    ini = 10
    step = 2
    num = 5
    boardnum = 40000
    traindataset = gamegen_policy(boardnum)
    validataset = gamegen_policy(boardnum // 4)
    hiddenlist = [ini + step * i for i in range(num)]
    trainerrs = [[] for i in range(num)]
    valierrs = [[] for i in range(num)]
    airec = [[] for i in range(num)]
    if number is None:
        itr = count()
    else:
        itr = range(number)
    for i in itr:
        try:
            with t:
                for i, hidden in enumerate(hiddenlist):
                    ai = ann_ai.ann_ai(pol_hidden=hidden, feature=['board', 'nextplayer'])
                    minres = ai.train(traindataset)
                    trainerrs[i].append(minres.fun)
                    cost = ai.getcost(validataset)
                    valierrs[i].append(cost)
                    airec[i].append(ai)
                    print('hidden layer %d done in %s' % (hidden, t()))
                print('\a')
        except KeyboardInterrupt:
            break
    trainerrmean = [mean(thiserr) for thiserr in trainerrs]
    validateerrmean = [mean(thiserr) for thiserr in valierrs]
    p1 = plt.plot([ini + step * i for i in range(num)], trainerrmean)
    p2 = plt.plot([ini + step * i for i in range(num)], validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    plt.show()
    interact(local=locals())


if __name__ == '__main__':
    r4_2d()
    #r8(['board'], ['board', 'winpt'], ['board', 'nextplayer'])
