'''
AI using artificial neural network.
'''
from numpy import array
from itertools import chain

from ttthelper import emptyspace_pos, gamegen
from learningfunc import ann
from base_ai import base_ai
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
import cythonann

'''
TODO:
Reinforcement learning using trained value network
Policy Network Training using perfectdataset
'''
class ann_ai(base_ai):
    def __init__(self, val_hidden=None, pol_hidden=None, feature=['board']):
        self.USE_VAL = True
        self.USE_POL = False
        self.feature = feature
        if self.USE_VAL:
            val_layernum = [self.feature_num] + ([] if val_hidden is None else val_hidden) + [1]
            self.val_ann = cythonann.ann(val_layernum)
        if self.USE_POL:
            pol_layernum = [self.feature_num] + ([] if pol_hidden is None else pol_hidden) + [1]
            self.pol_ann = cythonann.ann(pol_layernum)

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
            out = self.pol_ann.get(self.featureize_final(board))
            maxindex = out.index(max(out))
            return maxindex // 3, maxindex % 3
        else:
            raise NotImplementedError

    def process_dataset(self, dataset):
        data, ans = dataset
        data = array([i for i in chain.from_iterable([self.featureize_in_piece(d) for d in data])])
        ans = array([array([a]) for a in ans])
        return data, ans

    def train(self, dataset=None, pt=True, pt_option='all', gtol=1e-5):
        if dataset is None:
            data, ans = self.process_dataset(gamegen(game=100))
        else:
            data, ans = self.process_dataset(dataset)
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            minres = self.val_ann.train(data, ans, gtol)
            if pt:
                available_option = ['fun', 'hess_inv', 'jac', 'message', 'nfev', 'nit', 'njev', 'status', 'success', 'x']
                if pt_option == 'all':
                    print(minres)
                else:
                    for o in available_option:
                        if o in pt_option:
                            print('    %s:' % o, eval('minres.' + o))
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
