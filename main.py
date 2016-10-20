import logreg_ai
import ttttester
from statistics import mean, stdev, variance

'''
Datas:
Feature     Mean    StdDev
RANDOM      0.91    0.03
board       1.17    0.10
abs         0.93    0.08
'''
if __name__ == '__main__':
    # ai = logreg_ai().startlearn(game=100, opponent='random', pt=False, graph=False)
    # complete_check(algorithm=ai.getstep, pt=True)
    # complete_check(algorithm=randomstep, pt=True)
    TRY = 10
    scorerec = []
    try:
        for i in range(TRY):
            if i % min(10, TRY//10) == 0 and i:
                print('%d ais checked..' % i)
            ai = logreg_ai.logreg_ai(feature=['board'])
            # ai = ann_ai.ann_ai(layernum=[9, 18, 9])
            ai.startlearn(game=100, opponent='random', pt=False, graph=False)
            score = ttttester.complete_check(ai.getstep, pt=False)
            scorerec.append(score)
        print('Finished check.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(scorerec), stdev(scorerec)))
    except KeyboardInterrupt:
        print('Check interrupted with %d try.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (i, mean(scorerec), stdev(scorerec)))
