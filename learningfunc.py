from numpy import array, dot, log, e


def sigmoid(z):
    return 1/(1+e**(-z))


def costfunc(theta, data, ans):
    return sum(ans * -log(sigmoid(dot(data, theta))) - (1 - ans) * log(1 - sigmoid(dot(data, theta))))


def costfunc_d(theta, data, ans):
    return dot(data.T, (sigmoid(dot(data, theta)) - ans)) / len(theta)
