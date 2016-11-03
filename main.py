'''
Main script for this project.

The script does the following:
1. Create an instance of ai and train it. Class for AI is defined in logreg_ai.py and ann_ai.py
2. Check the ai's performance using ttttester.completecheck. The performance measurement is AEP(see ttttester.py)
3. Repeat if specified.
4. Opens an interactive console which helps manually checking ai trained.
'''
import logreg_ai
import ttttester
import ann_ai
from statistics import mean, stdev
from itertools import count
import json
import time
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

def main(game):
    scorerec = []
    TOL = 1e-5
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
            print('gradient tolerance:', TOL)
        if game <= 0:
            itr = count()
            print('Game will proceed until interrupt')
        else:
            itr = range(game)
            print('%d games will be played' % game)
        print('current time is %s.' % time.strftime('%X %x'))
        for i in itr:
            print('Checking %dth ai..' % (i + 1))
            if logreg:
                ai = logreg_ai.logreg_ai(feature=FEATURE)
                ai.startlearn(game=100, opponent='random', pt=False, graph=False)
            elif ann:
                ai = ann_ai.ann_ai(feature=FEATURE, val_hidden=VAL_HIDDEN)
                ai.train(dataset=dataset, pt=True, pt_option='all', gtol=TOL)
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
        print('Finished check.')
        try:
            print('Average AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (mean(scorerec), stdev(scorerec)))
        except:  # Less than 2 record will raise StatisticsError
            pass
    except KeyboardInterrupt:
        print('Check interrupted with %d try.\nAverage AEP for this ai is: %f\nStandard deviation for AEP is:%f' % (i, mean(scorerec), stdev(scorerec)))
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('current time is %s.' % time.strftime('%X %x'))
        if mscore > 0:
            a = ann_ai.ann_ai(feature=FEATURE)
            a.val_ann.theta = mtheta
            print('The best theta value is plugged into ann_ai object "a"')
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        try:
            __import__('code').interact(local=locals())
        except KeyboardInterrupt:
            pass


def handle_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', type=int, default=1, help='number of game')
    args = parser.parse_args()
    return (args.game,)

if __name__ == '__main__':
    # from cProfile import run
    # run('main()', 'profileresult-3')
    main(*handle_args())
