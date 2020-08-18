import winreg, requests
import numpy as np

from baselines.common.runners import AbstractEnvRunner
from time import sleep
from subprocess import Popen, DEVNULL

class Runner(AbstractEnvRunner):

    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, lam, mvs):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.mvs = mvs

    def start(self, render, n_attempts=25):
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

    def run(self, eval=False, render=False):

        # start mevea simulator and reset frontends

        proc = self.start(render)
        self.obs[:] = self.env.reset()

        # init experience minibatches

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        # act in environments step by step

        scores = [[] for _ in range(self.env.num_envs)]
        steps = [0 for _ in range(self.env.num_envs)]
        for _ in range(self.nsteps):

            # given observations, get action value and neglopacs

            if eval:
                actions, values, self.states, neglogpacs = self.model.eval_step(self.obs, S=self.states, M=self.dones)
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # take actions and check results

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for i in range(len(infos)):
                if 'r' in infos[i].keys():
                    scores[i].append(infos[i]['r'])
                if 'l' in infos[i].keys() and infos[i]['l'] > steps[i]:
                    steps[i] = infos[i]['l']
            mb_rewards.append(rewards)

        for i in range(self.env.num_envs):
            epinfos.append({'r': np.mean(scores[i]), 'l': steps[i]})

        # stop the simulator

        self.stop(proc)

        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

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
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])