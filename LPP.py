import numpy as np
def rbf(dist, t=1.0):
    return np.exp(-(dist / t))

def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist

def cal_rbf_dist(data, n_neighbors=10, t=1):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

def lpp(data, n_neighbors=30, t=1.0):
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)
    for i in range(N):
        D[i, i] = np.sum(W[i])
    L = D - W
    return L

def output_L(X,nidm):
    dist = cal_pairwise_dist(X)
    max_dist = np.max(dist)
    L = lpp(X,n_neighbors=nidm, t=0.01 * max_dist)
    return L