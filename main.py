from logreg_ai import logreg_ai
from ttttester import complete_check

ai = logreg_ai().startlearn(game=100, opponent='random')
complete_check(algorithm=ai.getstep, pt=True)
