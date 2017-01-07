'''
AI using artificial neural network.
'''
from numpy import array
from itertools import chain

import ttthelper as helper
from base_ai import base_ai
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
import cythonann
import learningfunc

'''
TODO:
Reinforcement learning using trained value network
Policy Network Training using perfectdataset
'''

INT = "__import__('code').interact(local=locals())"
class ann_ai(base_ai):
    def __init__(self, val_hidden=None, pol_hidden=None, feature=['board'], cython=False):
        if val_hidden is None and pol_hidden is None:
            raise ValueError('No ANN is used') 
        self.USE_VAL = False if val_hidden is None else True
        self.USE_POL = False if pol_hidden is None else True
        self.feature = feature
        if self.USE_VAL:
            val_layernum = [self.feature_num] + ([] if val_hidden is None else val_hidden) + [1]
            if cython:
                self.val_ann = cythonann.ann(val_layernum)
            else:
                self.val_ann = learningfunc.ann(val_layernum)
        elif self.USE_POL:
            pol_layernum = [self.feature_num] + ([] if pol_hidden is None else pol_hidden) + [9]
            if cython:
                self.pol_ann = cythonann.ann(pol_layernum)
            else:
                self.pol_ann = learningfunc.ann(pol_layernum)
        else:
            raise NotImplementedError

    #@profile
    def getstep(self, board, ainum, step):
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            mi, mj, mout = 0, 0, -10000 if ainum % 2 else 10000
            for nextboard, i, j in helper.emptyspace_pos(board, step):
                nextboard = self.featureize_final(nextboard)
                out = self.val_ann.get(nextboard)[0]  # An output's first one(while there's only one as it's val)
                if ainum % 2 == 1 and out > mout:
                    mi, mj, mout = i, j, out
                elif ainum % 2 == 0 and out < mout:
                    mi, mj, mout = i, j, out
            return mi, mj
        elif self.USE_POL:
            out = self.pol_ann.get(self.featureize_final(board))
            maxout, maxindex = 0, 0
            flatbd = helper.flatten(board)
            for i, o in enumerate(out):
                if flatbd[i] != 0:
                    continue
                if o > maxout:
                    maxout = o
                    maxindex = i
            return maxindex // 3, maxindex % 3

    #@profile
    def dataset_featureize(self, dataset):
        data, ans = dataset
        data = array([self.featureize_final(d) for d in data])
        ans = array([[a] for a in ans])
        return data, ans

    def getcost(self, dataset):
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            data, ans = self.dataset_featureize(dataset)
            return self.val_ann.costfunc(data, ans)
        elif self.USE_POL:
            raise NotImplementedError

    #@profile
    def train(self, dataset=None, pt=False, pt_option='all', gtol=1e-5):
        if self.USE_VAL and self.USE_POL:
            raise NotImplementedError
        elif self.USE_VAL:
            if dataset is None:
                dataset = helper.gamegen(gamenum=1000)
            data, ans = self.dataset_featureize(dataset)
            minres = self.val_ann.train(data, ans, gtol)
            if pt:
                available_option = ['fun', 'hess_inv', 'jac', 'message', 'nfev', 'nit', 'njev', 'status', 'success', 'x']
                if pt_option == 'all':
                    print(minres)
                else:
                    for o in available_option:
                        if o in pt_option:
                            print('    %s:' % o, eval('minres.' + o))
            return minres
        elif self.USE_POL:
            if dataset is None:
                raise ValueError
            data, ans = dataset
            minres = self.pol_ann.train(data, ans, gtol)