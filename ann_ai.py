from numpy import array
from itertools import chain

from ttthelper import emptyspace_pos, gamegen
from learningfunc import ann
from base_ai import base_ai


class ann_ai(base_ai):
    def __init__(self, val_layernum=None):
        if val_layernum is None:
            self.val_ann = ann([9, 9, 1])
        else:
            self.val_ann = ann(val_layernum)
        self.USE_VAL = True
        self.USE_POL = False
        self.feature = ['board']
        if self.USE_VAL and self.val_ann.layernum[-1] != 1:
            raise ValueError('val_ann has more than 1 output unit. Are you sure it is for value network?')

    def getstep(self, board, ainum, step):
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            mi, mj, mout = 0, 0, -10000 if ainum % 2 else 10000
            for nextboard, i, j in emptyspace_pos(board, step):
                nextboard = self.featureize_final(nextboard)
                out = self.val_ann.get(nextboard)[0]  # An output's first one(while there's only one as it's val)
                if ainum % 2 == 1 and out > mout:
                    mi, mj, mout = i, j, out
                elif ainum % 2 == 0 and out < mout:
                    mi, mj, mout = i, j, out
            return mi, mj
        elif self.USE_POL:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self, dataset=None, pt=True):
        if dataset is None:
            data, ans = gamegen(game=100)
        else:
            data, ans = dataset
        data = array([i for i in chain.from_iterable([self.featureize_in_piece(d) for d in data])])
        ans = array([array([a]) for a in ans])
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            minres = self.val_ann.train(data, ans)
            if pt:
                print(minres)
        elif self.USE_POL:
            raise NotImplementedError
        else:
            raise NotImplementedError

if __name__ == '__main__':
    a = ann_ai()
    inittheta = a.val_ann.theta
    a.train()
    input('Input to continue..')
    import ttttester
    ttttester.complete_check(a.getstep, pt=True)
    __import__('code').interact(local=locals())
