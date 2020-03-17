import pandas, sys

from matplotlib import pyplot as pp

if __name__ == '__main__':

    if len(sys.argv) == 2:
        policy = sys.argv[1]
    else:
        print('Specify policy folder.')
        sys.exit(1)

    fpath = 'policies/{0}/progress.csv'.format(policy)
    p = pandas.read_csv(fpath, delimiter=',')
    keys = p.keys()
    upd_idx = [i for i,key in enumerate(keys) if 'nupdates' in key][0]
    rew_idx = [i for i,key in enumerate(keys) if 'eprewmean' in key][0]
    max_rew_idx = [i for i, key in enumerate(keys) if 'eprewmax' in key][0]
    min_rew_idx = [i for i, key in enumerate(keys) if 'eprewmin' in key][0]
    exp_var_idx = [i for i, key in enumerate(keys) if 'explained_variance' in key][0]
    vals = p.values

    pp.plot(vals[:, upd_idx], vals[:, rew_idx], 'b')
    pp.errorbar(vals[:, upd_idx], vals[:, rew_idx], [vals[:, rew_idx] - vals[:, min_rew_idx], vals[:, max_rew_idx] - vals[:, rew_idx]], fmt='ok', color='blue', ecolor='blue', lw=3)
    pp.show()

