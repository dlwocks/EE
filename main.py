import logreg_ai
import ttttester
import ann_ai
import ttthelper
from statistics import mean, stdev, variance
from itertools import count
import json
'''
Datas:
Feature     Mean    StdDev
RANDOM      0.91    0.03
board       1.17    0.10
abs         0.93    0.08

VAL_ANN:
Feature/Hidden:
board + [] = 1.225722
'''

def main():
    TRY = 10
    scorerec = []
    try:
        # dataset = ttthelper.rndgen(game=100)
        logreg, ann = False, True
        if ann:
            with open('D:\\EE\\app\\rnddataset.dat') as o:
                dataset = json.load(o)
            print('data for ann is loaded from file.')
        mscore, mtheta = 0, None
        FEATURE = ['board']
        VAL_HIDDEN = [9]
        if logreg:
            print('current logreg has feature of', FEATURE)
        if ann:
            print('current ann has feature of', FEATURE, 'and hidden layer of', VAL_HIDDEN)
        #itr = count()
        itr = range(1)
        for i in itr:
            if i % (i // 10 + 1) == 0:
                print('Checking %dth ai..' % (i + 1))
            if logreg:
                ai = logreg_ai.logreg_ai(feature=FEATURE)
                ai.startlearn(game=100, opponent='random', pt=False, graph=False)
            elif ann:
                ai = ann_ai.ann_ai(feature=FEATURE, val_hidden=VAL_HIDDEN)
                ai.train(dataset=dataset, pt=True, pt_option=['message', 'fun'])
            score = ttttester.complete_check(ai.getstep, pt=False)
            print('    score:', score)
            scorerec.append(score)
            if score > mscore:
                print('Maximum value found on %dth attempt: %f' % (i + 1, score))
                mscore = score
                if logreg:
                    mtheta = ai.theta_value
                elif ann:
                    mtheta = ai.val_ann.theta
        # print('Finished check.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(scorerec), stdev(scorerec)))
    except KeyboardInterrupt:
        print('Check interrupted with %d try.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (i, mean(scorerec), stdev(scorerec)))
    except:
        import traceback
        traceback.print_exc()
    finally:
        if mscore > 0:
            a = ann_ai.ann_ai(feature=FEATURE)
            a.val_ann.theta = mtheta
            print('The best theta value is plugged into ann_ai object "a"')
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        try:
            __import__('code').interact(local=locals())
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    # from cProfile import run
    # run('main()', 'profileresult-3')
    main()
