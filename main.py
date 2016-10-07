from logreg_ai import logreg_ai
from ai import algorithm_wiki
from ttttester import complete_check, randomstep
from statistics import mean, stdev

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
            time = complete_check(logreg_ai().startlearn(game=100, opponent='random', pt=False, graph=False).getstep, pt=False)
            timerec.append(time)
        print('Finished check.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(timerec), stdev(timerec)))
    except KeyboardInterrupt:
        print('Check interrupted.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(timerec), stdev(timerec)))
