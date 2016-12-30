@profile
def test(num):
    # Test a logreg_ai
    from logreg_ai import logreg_ai
    from ann_ai import ann_ai
    import ttttester
    import ttthelper
    lrai = logreg_ai()
    lrai.startlearn(game=num, pt=False, graph=False)
    # Train logreg_like ann in the logreg way
    dataset = [[], []]
    ai = ann_ai(val_hidden=[], cython=False)
    for i in range(num):
        subdataset = ttthelper.gamegen(1, algs=[ai.getstep, ttthelper.randomstep])
        dataset[0] += subdataset[0]
        dataset[1] += subdataset[1]
        ai.train(dataset=dataset, pt=False)

test(100)