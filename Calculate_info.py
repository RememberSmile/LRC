import  numpy as np
def Calculate_K(W_f):
    agg_k = []
    result = np.where(W_f < 0, 0, W_f)
    for i in range(result.shape[0]):
        cout = 0
        for j in range(result.shape[1]):
            if result[j, i] > 0:
                # if i != j:
                    cout = cout + 1
        agg_k.append(cout)
    return agg_k
def Calculate_Index(W_f):
    agg_index = []
    result = np.where(W_f < 0, 0, W_f)
    for i in range(result.shape[0]):
        relevance = []
        for j in range(result.shape[1]):
            if result[j, i] > 0:
                # if i!=j:
                    relevance.append(j)
        agg_index.append(relevance)
    return agg_index
def Calculate_W(W_f):
    agg_w = []
    result = np.where(W_f < 0, 0, W_f)
    for i in range(result.shape[0]):
        cout_w = 0
        for j in range(result.shape[1]):
            if result[j, i] > 0:
                # if i!=j:
                    cout_w = result[j,i] + cout_w
        agg_w.append(cout_w)
    return agg_w
