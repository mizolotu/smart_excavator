import pandas

from matplotlib import pyplot as pp
from scipy.signal import savgol_filter


if __name__ == '__main__':

    fpath = 'policies/human+ppo/progress.csv'
    p = pandas.read_csv(fpath, delimiter=',')
    keys = p.keys()
    upd_idx = [i for i,key in enumerate(keys) if 'nupdates' in key][0]
    rew_idx = [i for i,key in enumerate(keys) if 'eprewmean' in key][0]
    max_rew_idx = [i for i, key in enumerate(keys) if 'eprewmax' in key][0]
    min_rew_idx = [i for i, key in enumerate(keys) if 'eprewmin' in key][0]
    exp_var_idx = [i for i, key in enumerate(keys) if 'explained_variance' in key][0]
    vals = p.values

    pp.plot(vals[:, upd_idx], vals[:, rew_idx])
    pp.show()

