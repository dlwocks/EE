from logreg_ai import logreg_ai
from ai import algorithm_wiki
from ttttester import complete_check, randomstep
from statistics import mean, stdev, variance

'''
Datas:
Feature     Mean    StdDev
RANDOM      0.91    0.03
board       1.17    0.10
abs         0.93    0.08
'''
if __name__ == '__main__':
    #ai = logreg_ai().startlearn(game=100, opponent='random', pt=False, graph=False)
    #complete_check(algorithm=ai.getstep, pt=True)
    # complete_check(algorithm=randomstep, pt=True)
    TRY = 100
    timerec =[]
    try:
        for i in range(TRY):
            if i % 5 == 0 and i:
                print('%d ais checked..' % i)
            time = complete_check(logreg_ai(feature=['board']).startlearn(game=100, opponent='random', pt=False, graph=False).getstep, pt=False)
            # time = complete_check(randomstep, pt=False)
            timerec.append(time)
        print('Finished check.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f\nVariance:%f' % (mean(timerec), stdev(timerec), variance(timerec)))
    except KeyboardInterrupt:
        print('Check interrupted with %d try.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f\nVariance:%f' % (i, mean(timerec), stdev(timerec), variance(timerec)))
