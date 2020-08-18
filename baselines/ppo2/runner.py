import numpy as np
import tensorflow as tf
import winreg, requests

from baselines.common.runners import AbstractEnvRunner
from time import sleep
from subprocess import Popen, DEVNULL

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam, mvs):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.mvs = mvs

    def start(self, render, n_attempts=25, delay=5.0):
        http_url = 'http://127.0.0.1:5000/id'
        regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
        (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
        solverpath += r'\Bin\MeveaSolver.exe'
        winreg.CloseKey(regkey)
        if render:
            solver_args = [solverpath, r'/mvs', self.mvs]
        else:
            solver_args = [solverpath, r'/headless', r'/mvs', self.mvs]
        proc = []
        for i in range(self.env.num_envs):
            sleep(delay)
            print('Trying to start solver {0}...'.format(i))
            ready = False
            while not ready:
                assigned = False
                attempt = 0
                sleep(i)
                solver_proc = Popen(solver_args, stderr=DEVNULL, stdout=DEVNULL)
                while not assigned:
                    try:
                        j = requests.get(http_url, json={'id': i}).json()
                        assigned = j['assigned']
                    except Exception as e:
                        print('Exception when assigning id:')
                        print(e)
                    attempt += 1
                    if attempt >= n_attempts:
                        break
                    sleep(1.0)
                if assigned:
                    ready = True
                    print('Solver {0} has successfully started!'.format(i))
                    proc.append(solver_proc)
                else:
                    print('Could not start solver {0} :( Trying again...'.format(i))
                    solver_proc.terminate()
        return proc

    def stop(self, proc):
        http_url = 'http://127.0.0.1:5000/id'
        for pi, p in enumerate(proc):
            assigned = True
            while assigned:
                try:
                    j = requests.post(http_url, json={'id': pi}).json()
                    assigned = j['assigned']
                except Exception as e:
                    print('Exception when reseting id:')
                    print(e)
            p.terminate()

    def run(self, render=False):

        # start mevea simulator and reset frontends

        proc = self.start(render)
        self.obs[:] = self.env.reset()

        # Here, we init the lists that will contain the mb of experiences

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        scores = [[] for _ in range(self.env.num_envs)]
        masses = [[] for _ in range(self.env.num_envs)]
        dists = [[] for _ in range(self.env.num_envs)]

        # For n in range number of steps

        for _ in range(self.nsteps):

            obs = tf.constant(self.obs)
            actions, values, self.states, neglogpacs = self.model.step(obs)
            actions = actions._numpy()
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values._numpy())
            mb_neglogpacs.append(neglogpacs._numpy())
            mb_dones.append(self.dones)

            # Take actions in env and look the results

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for i in range(len(infos)):
                if 'r' in infos[i].keys():
                    scores[i].append(infos[i]['r'])
                if 'm' in infos[i].keys():
                    masses[i].append(infos[i]['m'])
                if 'd' in infos[i].keys():
                    dists[i].append(infos[i]['d'])
            mb_rewards.append(rewards)

        for i in range(self.env.num_envs):
            epinfos.append({'r': np.mean(scores[i]), 'm': np.mean(masses[i]), 'd': np.mean(dists[i])})

        # stop the simulator

        self.stop(proc)

        # Batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(tf.constant(self.obs))._numpy()

        # discount/bootstrap off value fn

        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
