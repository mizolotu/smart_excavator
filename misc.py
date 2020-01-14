import numpy as np

def moving_average(x, step=2, window=2):
    seq = []
    n = x.shape[0]
    for i in np.arange(0, n, step):
        idx = np.arange(np.maximum(0, i - window), np.minimum(n - 1, i + window + 1))
        seq.append(np.mean(x[idx, :], axis=0))
    return np.vstack(seq)