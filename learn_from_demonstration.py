import pickle
import numpy as np
from matplotlib import pyplot as pp

def retrieve_original_dataset(data_file='data/level_points.pkl', t_window=1):
    with open(data_file, 'rb') as f:
        levels = pickle.load(f)
    data = []
    t_last = np.inf
    for li,level in enumerate(levels):
        t = level[1][:, 0]
        x = level[0]
        if t[1] < t_last:
            t_last = t[-1]
            data.append([])
        t_line = np.arange(t[0], t[-1], t_window)
        m = len(t_line)
        n = x.shape[1]
        p = np.zeros((m, n))
        for j in range(x.shape[1]):
            p[:, j] = np.interp(t_line, t, x[:, j])
        data[-1].append(p)
    return data

def augment_data(sample, a_max=205):
    sample_aug = []
    dig_angle_orig = np.max([np.max(d[:, 0]) for d in sample])
    dig_angle_new = np.random.rand() * a_max
    alpha = dig_angle_new / dig_angle_orig
    for j in range(len(data_orig)):
        a = sample[j][:, 0:1]
        x = sample[j][:, 1:]
        a_new = a * alpha
        sample_aug.append(np.hstack([a_new, x]))
    return sample_aug

if __name__ == '__main__':
    data_orig = retrieve_original_dataset()
    sample_orig = np.random.choice(data_orig)
    dsa = augment_data(sample_orig)