from logreg_ai import logreg_ai
from ai import algorithm_wiki
from ttttester import complete_check, randomstep

if __name__ == '__main__':
    #ai = logreg_ai().startlearn(game=100, opponent='random', pt=False, graph=False)
    #complete_check(algorithm=ai.getstep, pt=True)
    # complete_check(algorithm=randomstep, pt=True)
    TRY = 10
    print(sum([complete_check(logreg_ai().startlearn(game=100, opponent='random', pt=False, graph=False).getstep, pt=False) for _ in range(TRY)])/TRY)
