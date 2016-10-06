from numpy import array, dot, log, e


def sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    if not all(sigmoid(dot(data, theta))):
        raise ValueError('theta: %s\ndata:%s\nans:%s' % (repr(theta),repr(data),repr(ans)))
    sig = sigmoid(dot(data, theta))
    return sum(ans * -log(sig) - (1 - ans) * log(1 - sig))


def costfunc_d(theta, data, ans):
    if not all(sigmoid(dot(data, theta))):
        raise ValueError
    return dot(data.T, (sigmoid(dot(data, theta)) - ans)) / len(theta)
