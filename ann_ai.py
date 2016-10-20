from learningfunc import ann
from base_ai import base_ai


class ann_ai(base_ai):
    def __init__(self, layernum):
        self.ann = ann(layernum)

    def getstep(self, board, ainum, step):
        returned = self.ann.fowardprop(board)
        raise NotImplementedError

    def startlearn(self, game='converge', difftol=0.01, opponent='random', graph=True, pt=True):
        raise NotImplementedError
