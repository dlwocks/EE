from numpy import array

from learningfunc import gen_piece
import features

'''
Use of:     Value Network   Policy Network  Tree Search
Current:    Yes(ML)         None            depth-1
Desired:    None            Yes(ML)         depth-0
'''


class base_ai(object):
    FEATURE_FUNC_MAP = {'board': features.board}
    FEATURE_NUM_MAP = {'board': 9}

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
