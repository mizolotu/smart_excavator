import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

def learn(network, env, nsteps, total_timesteps, mvs, ckpt,
    seed=None,
    ent_coef=0.0,
    lr=1e-3,
    vf_coef=0.5,
    max_grad_norm=0.5,
    noptepochs=4,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    cliprange=0.2,
    save_interval=1,
    model_fn=None,
    update_fn=None,
    init_fn=None,
    mpi_rank_weight=1,
    comm=None,
    load_path=None,
    **network_kwargs):

    set_global_seeds(seed)
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # build policy

    policy = build_policy(env, network, **network_kwargs)

    # Calculate the batch_size

    nenvs = env.num_envs
    nminibatches = 1
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)

    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(
        policy=policy,
        nbatch_act=nenvs,
        nbatch_train=None, #nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        comm=comm,
        mpi_rank_weight=mpi_rank_weight
    )

    if load_path is not None:
        model.load(load_path)
        print('Model has been successfully loaded from {0}'.format(load_path))
    else:
        try:
            lp = osp.join(logger.get_dir(), 'checkpoints/{0}'.format(ckpt))
            model.load(lp)
            print('Model has been successfully loaded from {0}'.format(lp))
        except Exception as e:
            print(e)

    # Instantiate the runner object and episode buffer

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, mvs=mvs)
    epinfobuf = deque(maxlen=log_interval*nenvs)
    best_reward = -np.inf

    if init_fn is not None:
        init_fn()

    # Start total timer

    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates # decreases from 1 to 0
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.

        mblossvals = []
        if states is None: # nonrecurrent version

            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version

            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    # print(states.shape, mbstates.shape)
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.perf_counter()
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("misc/fps", fps)

            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('stats/eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('stats/eprewmin', np.min([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('stats/eprewmax', np.max([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('stats/eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('misc/' + lossname, lossval)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, 'last')
            print('Saving to', savepath)
            model.save(savepath)
            if len(epinfobuf) >= log_interval and safemean([epinfo['r'] for epinfo in epinfobuf]) > best_reward:
                savepath = osp.join(checkdir, 'best')
                print('Saving to', savepath)
                model.save(savepath)
                best_reward = safemean([epinfo['r'] for epinfo in epinfobuf])

    model.sess.close()
    return model

def demonstrate(network, env, nsteps, mvs, ckpt,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    mpi_rank_weight=1,
    comm=None,
    gamma=0.99,
    lam=0.95,
    load_path=None
):

    policy = build_policy(env, network)

    model = Model(
        policy=policy,
        nbatch_act=1,
        nbatch_train=None,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        comm=comm,
        mpi_rank_weight=mpi_rank_weight
    )

    if load_path is not None:
        model.load(load_path)
        print('Model has been successfully loaded from {0}'.format(load_path))
    else:
        try:
            lp = osp.join(logger.get_dir(), 'checkpoints/{0}'.format(ckpt))
            model.load(lp)
            print('Model has been successfully loaded from {0}'.format(lp))
        except Exception as e:
            print(e)
            print('No model has been loaded. Neural network with random weights is used.')

    # Instantiate the runner object and episode buffer

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, mvs=mvs)
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(render=True)

    print('Demo completed! Reward: {0}'.format(epinfos[0]['r']))
    print('\nPress Ctrl+C to stop the demo...')

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)