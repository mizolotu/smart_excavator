import time
import numpy as np
import tensorflow as tf
import os.path as osp

from baselines.ppo2.model import Model
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.models import get_network_builder
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

def learn(network, env, nsteps, total_timesteps, mvs,
        seed=None, ent_coef=0.0, lr=3e-4,
        vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
        log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
        load_path=None, model_fn=None, **network_kwargs
    ):

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env

    nenvs = env.num_envs

    # Get state_space and action_space

    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        network = policy_network_fn(ob_space.shape)

    # Calculate the batch_size

    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(ac_space=ac_space, policy_network=network, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm)

    if load_path is None:
        load_path = osp.join(logger.get_dir(), 'checkpoints')
    load_path = osp.expanduser(load_path)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=1)
    if manager.latest_checkpoint is not None:
        print('Restoring model from {0}'.format(manager.latest_checkpoint))
        ckpt.restore(manager.latest_checkpoint)

    # Instantiate the runner object

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, mvs=mvs)
    epinfobuf = deque(maxlen=log_interval*nenvs)

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.

        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (tf.constant(arr[mbinds]) for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            raise ValueError('Not Support Yet')

        # Feedforward --> get losses --> update

        lossvals = np.mean(mblossvals, axis=0)

        # End timer

        tnow = time.perf_counter()
        if update % log_interval == 0 or update == 1:

            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)

            ev = explained_variance(values, returns)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmin', np.min([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmax', np.max([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('soilmassmean', safemean([epinfo['m'] for epinfo in epinfobuf]))
            logger.logkv('soilmassmin', np.min([epinfo['m'] for epinfo in epinfobuf]))
            logger.logkv('soilmassmax', np.max([epinfo['m'] for epinfo in epinfobuf]))
            logger.logkv('distmean', safemean([epinfo['d'] for epinfo in epinfobuf]))
            logger.logkv('distmin', np.min([epinfo['d'] for epinfo in epinfobuf]))
            logger.logkv('distmax', np.max([epinfo['d'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

            # save checkpoint

            manager.save()

    return model

def demonstrate(network, env, nsteps, mvs, load_path,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    **network_kwargs
):

    ob_space = env.observation_space
    ac_space = env.action_space

    policy_network_fn = get_network_builder(network)(**network_kwargs)
    network = policy_network_fn(ob_space.shape)

    model = Model(
        ac_space=ac_space,
        policy_network=network,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm
    )

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print('Model has been successfully loaded from {0}'.format(load_path))
    else:
        print('No model has been loaded. Neural network with random weights is used.')

    # Instantiate the runner object and episode buffer

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, mvs=mvs)
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(render=True)

    print('Demo completed! Reward: {0}'.format(epinfos[0]['r']))
    print('\nPress Ctrl+C to stop the demo...')

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


