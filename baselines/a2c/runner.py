import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        #self.batch_action_shape = [env.num_envs * nsteps, 1]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):

        self.obs = self.env.reset()

        # We initialize the lists that will contain the mb of experiences

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        epinfos = []
        scores = [[] for _ in range(self.env.num_envs)]
        steps = [0 for _ in range(self.env.num_envs)]
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            obs, rewards, dones, infos = self.env.step(actions)

            for i in range(len(infos)):
                if 'r' in infos[i].keys():
                    scores[i].append(infos[i]['r'])
                if 'l' in infos[i].keys() and infos[i]['l'] > steps[i]:
                    steps[i] = infos[i]['l']
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        for i in range(self.env.num_envs):
            epinfos.append({'r': np.mean(scores[i]), 'l': steps[i]})
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()

            for n, (rewards, dones, values, last_value, last_done) in enumerate(zip(mb_rewards, mb_dones, mb_values, last_values, self.dones)):

                #rewards = rewards.tolist()
                #dones = dones.tolist()
                #if dones[-1] == 0:
                #    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                #else:
                #    rewards = discount_with_dones(rewards, dones, self.gamma)

                advs = np.zeros(self.nsteps)
                lastgaelam = 0
                for t in reversed(range(self.nsteps)):
                    if t == self.nsteps - 1:
                        nextnonterminal = 1.0 - last_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advs[t] = lastgaelam = delta + self.gamma * 0.95 * nextnonterminal * lastgaelam
                rewards = advs + values

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
