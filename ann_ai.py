from learningfunc import ann


class ann_ai(object):
    def __init__(self, layernum):
        self.ann = ann(layernum)

    def getstep(self, board, ainum, step):
        raise NotImplementedError

    def startlearn(self, game='converge', difftol=0.01, opponent='random', graph=True, pt=True):
        raise NotImplementedError
