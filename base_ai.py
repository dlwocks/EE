'''
The base of ann_ai and logreg_ai
'''
from numpy import array

from learningfunc import gen_piece
import features as ft

'''
Use of:     Value Network   Policy Network  Tree Search
Current:    Yes(ML)         None            depth-1
Desired:    None            Yes(ML)         depth-0
'''


class base_ai(object):
    FEATURE_FUNC_MAP = {'board': ft.board,
                        'abs': ft.absboard,
                        'extbd': ft.extboard,
                        'nboard': ft.nboard,
                        'lboard': ft.lboard,
                        'oboard': ft.oboard,
                        'orboard': ft.orboard,
                        'ctsur': ft.ctsur,
                        'nextplayer': ft.nextplayer}
    FEATURE_NUM_MAP = {'board': 9,
                       'abs': 9,
                       'extbd': 18,
                       'nboard': 9,
                       'lboard': 9,
                       'oboard': 9,
                       'orboard': 9,
                       'ctsur': 18,
                       'nextplayer': 1}

    @property
    def feature_num(self):
        return sum([self.FEATURE_NUM_MAP[f] for f in self.feature])

    def featureize_in_piece(self, board):
        piece = gen_piece(list(array(board).reshape((9,))))
        data = []
        for p in piece:
            temp = []
            for f in self.feature:
                temp.extend(self.FEATURE_FUNC_MAP[f](p))
            data.append(temp)
        return array(data)

    def featureize_final(self, board):
        ret = []
        for f in self.feature:
            ret.extend(self.FEATURE_FUNC_MAP[f](list(array(board).reshape((9,)))))
        return array(ret)
