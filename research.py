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

def mean(ls, ignorelarge=0, ignoresmall=0):
    if len(ls) - ignorelarge - ignoresmall < 1:
        raise RuntimeError
    if not (ignoresmall or ignorelarge):
        return stat.mean([i for i in ls if not np.isnan(i)])
    ls.sort()
    return stat.mean([i for i in ls[ignoresmall: -ignorelarge] if not np.isnan(i)])

def stdev(ls):
    return stat.stdev(ls)


class timer(object):
    def __call__(self):
        return str(self._time)

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
def r1(gamenum=6000, hidden=[9]):  # thesre are the specific config.
    t = timer()
    if isinstance(hidden, int):
        hidden = [hidden]
    allptrec = []
    ptrec = []
    airec = []
    while True:
        try:
            with t:
                dataset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                ai.train(dataset, pt=False)
                pt = ttttester.complete_check(ai.getstep)
            print('(%s)' % t(), end=' ')
            allptrec.append(pt)
            threshold = 91
            if pt > threshold:
                print('pt larger than thershold: %0.2f. Recorded.' % pt)
                ptrec.append(pt)
                airec.append(ai)
            else:
                print('pt smaller than threshold: %0.2f.' % pt)
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
def r3(ini=0, step=1, num=20):  # inis for hidden layer
    t = timer()
    if ini == 0:
        hiddenslist = [[]] + [[ini + step * i] for i in range(1, num)]
    elif ini > 0:
        hiddenslist = [[ini + step * i] for i in range(num)]
    gamenumlist = [100, 500, 1000]
    ptrec = [[[] for _ in range(num)] for _ in range(len(gamenumlist))]
    while True:
        try:
            with t:
                for i, gamenum in enumerate(gamenumlist):
                    dataset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                    for j, hidden in enumerate(hiddenslist):
                        ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                        ai.train(dataset, pt=False)
                        pt = ttttester.complete_check(ai.getstep)
                        ptrec[i][j].append(pt)
            print('Done in %s' % t())
        except KeyboardInterrupt:
            break
    ptrecmean = [[mean(thisrec) for thisrec in gamerec] for gamerec in ptrec]
    for i, gamerecmean in enumerate(ptrecmean):
        plt.plot([ini + step * i for i in range(num)], gamerecmean)
        plt.title()
    plt.show()
    interact(local=locals())


'''
I haven't considered to measure the error of ann directly. AEP is not an accurate measure on the accuracy of the ann
Following script will directly measure the error of ann to **choose the number of hidden layer**.
It is similar to r3, but use error of ann on **validation set** as measure.
Value Network specific.

One Result: 14.8?
'''
def r4(ini=0, step=1, num=6):
    t = timer()
    hiddenslist = [ini + step * i for i in range(num)]
    trainerrs = [[] for i in range(num)]
    validateerrs = [[] for i in range(num)]
    trainnum = 1000
    while True:
        try:
            with t:
                trainset = ttthelper.gamegen(trainnum, algs=[ttthelper.randomstep] * 2)
                validateset = ttthelper.gamegen(trainnum//4, algs=[ttthelper.randomstep] * 2)
                for i, hidden in enumerate(hiddenslist):
                    ai = ann_ai.ann_ai(val_hidden=hidden, pol_hidden=None, feature=['board'])
                    minres = ai.train(trainset, pt=False)
                    interact(local=locals())
                    trainerrs[i].append(minres.fun)
                    validateerrs[i].append(ai.getcost(validateset))
            print('Single: %s  Acc: %s  Num: %s' % (t.time, t.acc, t.num))
        except KeyboardInterrupt:
            break
    trainerrmean = [mean(thiserr)/4 for thiserr in trainerrs]
    validateerrmean = [mean(thiserr) for thiserr in validateerrs]
    p1 = plt.plot(hiddenslist, trainerrmean)
    p2 = plt.plot(hiddenslist, validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    coefs = np.polyfit([ini + step * i for i in range(num)], validateerrmean, 2)
    def poly(a, b, c, x):
        return a*(x**2) + b*x + c
    plt.plot(hiddenslist, [poly(*coefs, i) for i in hiddenslist])
    plt.show()
    interact(local=locals())

'''
The optimal number of hidden layer will be determined above.
Now this script will measure the error of ann to **choose the number of training set**.
It is similar to r2, but use error of ann on **validation set** as measure.
Value Network specific.
To Think: Should validateset be outside of gamenumlist loop?
'''
def r5(ini=1000, step=1000, num=10):
    OPTIMAL_LAYERNUM = 9
    t = timer()
    gamenumlist = [ini + step * i for i in range(num)]
    trainerrs = [[] for i in range(num)]
    validateerrs = [[] for i in range(num)]
    while True:
        try:
            with t:
                validateset = ttthelper.gamegen(ini//2, algs=[ttthelper.randomstep] * 2)
                for i, gamenum in enumerate(gamenumlist):
                    trainset = ttthelper.gamegen(gamenum, algs=[ttthelper.randomstep] * 2)
                    ai = ann_ai.ann_ai(val_hidden=[OPTIMAL_LAYERNUM], pol_hidden=None, feature=['board'])
                    minres = ai.train(trainset, pt=False)
                    trainerrs[i].append(minres.fun / gamenumlist[i] * ini / 2)  # 2 as from gamegen(ini//`2`)
                    validateerrs[i].append(ai.getcost(validateset))
            print('Single: %s   Acc: %s   Num: %s' % (t.time, t.acc, t.num))
        except KeyboardInterrupt:
            break
    trainerrmean = [mean(thiserr, ignorelarge=1) for thiserr in trainerrs]
    validateerrmean = [mean(thiserr, ignorelarge=1) for thiserr in validateerrs]
    p1 = plt.plot([ini + step * i for i in range(num)], trainerrmean)
    p2 = plt.plot([ini + step * i for i in range(num)], validateerrmean)
    plt.legend((p1[0], p2[0]), ('Train', 'Validate'))
    plt.show()
    interact(local=locals())


if __name__ == '__main__':
    r4()
