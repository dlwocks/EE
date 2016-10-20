from numpy import array

from learningfunc import gen_piece


class base_ai(object):
    def featureize_in_piece(self, board):
        piece = gen_piece(list(array(board).reshape((9,))))
        data = []
        for p in piece:
            temp = []
            for f in self.feature:
                temp.extend(self.FEATURE_FUNC_MAP[f](p))
            data.append(temp)
        return data

    def featureize_final(self, board):
        ret = []
        for f in self.feature:
            ret.extend(self.FEATURE_FUNC_MAP[f](list(array(board).reshape((9,)))))
        return ret