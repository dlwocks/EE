import logreg_ai
import ttttester
import ann_ai
import ttthelper
from statistics import mean, stdev, variance
from itertools import count
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
        dataset = ttthelper.rndgen(game=100)
        mscore, mtheta = 0, None
        for i in count():
            if i != 0:
                print('%d ais checked..' % i)
            # ai = logreg_ai.logreg_ai(feature=['board'])
            ai = ann_ai.ann_ai()
            ai.train(dataset=dataset, pt=False)
            score = ttttester.complete_check(ai.getstep, pt=False)
            scorerec.append(score)
            if score > mscore:
                print('Maximum value found on %dth attempt: %f' % (i + 1, score))
                mscore = score
                mtheta = ai.val_ann.theta
        print('Finished check.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(scorerec), stdev(scorerec)))
    except KeyboardInterrupt:
        print('Check interrupted with %d try.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (i, mean(scorerec), stdev(scorerec)))
    except:
        import traceback
        traceback.print_exc()
    finally:
        __import__('code').interact(local=locals())
