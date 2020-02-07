import numpy as np
import sys, os
import os.path as osp

from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common.vec_env import SubprocVecEnv
from train_ai import create_env

if __name__ == '__main__':

    # disable cuda if needed

    if'cpu' in sys.argv[1:]:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #  get data: 0 - index, 1,2,3,4 - target, 5 - time, 6 - mass, 7 - dumped, 8,... - cycle

    action_dim = 4
    other_dim = 4
    n_cycles = 4
    time_idx = 5
    mass_idx = 6
    rew_idx = 7
    fname = 'data/policy_data.txt'
    data = np.loadtxt(fname, dtype=float, delimiter=',')
    n_samples = data.shape[0]
    assert n_samples % n_cycles == 0
    n_series = n_samples // n_cycles
    sample_len = data.shape[1]
    n_labels = sample_len - action_dim - other_dim
    assert n_labels % action_dim == 0
    series_len = n_labels // action_dim
    n_features = 2 * action_dim + 1

    # reward coefficients

    m_coeff = 0.5
    d_coeff = 0.4
    t_coeff = 0.1

    # generate dataset

    x = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, n_labels))
    r = np.zeros((n_samples, 1))
    q = np.zeros((n_samples, 1))
    for i in range(n_series):
        v = 0
        series = data[i*n_cycles:(i+1)*n_cycles, :]
        dig_target = np.mean(series[:, 1:action_dim+1], axis=0)
        m_max = 0
        for j in range(n_cycles):
            idx = i * n_cycles + j
            x_dig = series[j, 1:action_dim+1]
            dist_to_target = np.linalg.norm(x_dig - dig_target)
            t_elapsed = series[j, time_idx]
            m_dumped = series[j, rew_idx]
            x[idx, :action_dim] = x_dig
            x[idx, action_dim:action_dim+action_dim] = dig_target
            x[idx, action_dim+action_dim] = m_max
            m_max = series[j, mass_idx]
            s = series[j, action_dim + other_dim:]
            y[idx, :] = - np.ones_like(s) + 2 * s  # rescale to [-1, 1] interval
            r[idx, 0] = m_coeff * m_dumped + t_coeff * (1 - t_elapsed) + d_coeff * (1 - dist_to_target)
            v += r[idx, 0]
            q[idx, 0] = v
    print('X shape: {0}, Y shape: {1}'.format(x.shape, y.shape))

    # create model

    nenvs = 2
    network = 'mlp'
    nminibatches = 4
    nsteps = 8
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    mpi_rank_weight = 1
    env_fns = [create_env(key) for key in range(nenvs)]
    env = SubprocVecEnv(env_fns)
    nbatch_train = nenvs * nsteps // nminibatches
    policy = build_policy(env, network)
    model = Model(
        policy=policy,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        comm=None,
        mpi_rank_weight=mpi_rank_weight
    )

    # train model

    lr = 1e-4
    epochs = int(1e6)
    idx = np.arange(n_samples)
    losses = []
    for e in range(epochs):
        np.random.shuffle(idx)
        obs = x[idx[:nbatch_train], :]
        action_means = y[idx[:nbatch_train], :]
        loss = model.train_on_demo(lr, obs, action_means)
        losses.append(loss[0])
        if e % (epochs // 100) == 0 and e > 0:
            print(obs)
            print('{0}% completed, loss: {1}'.format(e // (epochs // 100), np.mean(losses)))
            losses = []

    # save model

    checkdir = 'policies/human/checkpoints'
    savepath = osp.join(checkdir, 'last')
    print('Saving hOOman policy to {0}'.format(savepath))
    model.save(savepath)