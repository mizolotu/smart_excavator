import winreg, subprocess, gym, requests
import numpy as np

from time import sleep, time
from matplotlib import pyplot as pp

class ExcavatorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, id,
        mws='C:\\Users\\iotli\\PycharmProjects\\SmartExcavator\\mws\\env.mws',
        obs_dim=9,
        action_dim=4,
        http_url='http://127.0.0.1:5000',
        discrete_action=None,
    ):

        super(ExcavatorEnv, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # solver args

        regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
        (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
        solverpath += r'\Bin\MeveaSolver.exe'
        winreg.CloseKey(regkey)
        self.solver_args = [solverpath, r'/loadmws', mws, r'/saveplots', r'/silent']

        # id and counts

        self.episode_count = 0
        self.step_count = 0
        self.n_steps = 32
        self.n_series = 8
        self.step_count_max = self.n_series
        self.env_id = id
        self.backend_assigned = False
        self.dig_target = None
        self.emp_target = None
        self.http_url = http_url

        # state, action and reward coefficients

        self.mass = 0
        self.time_coeff = 0.1
        self.dist_coeff = 0.4
        self.mass_coeff = 0.5
        self.collision_coeff = 0.0
        self.mass_thr = 0.05
        self.x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
        self.x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
        self.m_max = 1000.0
        self.t_max = 60.0

        # observation and action spaces

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float)
        self.discrete_action = discrete_action
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim * self.n_steps,), dtype=np.float)
        self.debug = True

    def step(self, action, x_ind=4, m_ind=-1, x_thr=5.0):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        trajectory = action.reshape(self.n_steps, self.action_dim)
        pp.plot(trajectory)
        pp.show()
        self.mass = 0
        jdata = self._post_target()
        x_dig = (jdata['x'] - self.x_min) / (self.x_max - self.x_min)
        n_collisions = 0
        t_cycle_start = time()
        for i in range(trajectory.shape[0]):
            target = np.clip(self.x_min + (self.x_max - self.x_min) * trajectory[i, :], self.x_min, self.x_max)
            print(target)
            jdata = self._post_target(target)
            n_collisions += jdata['c']
            mass = jdata['m'] / self.m_max
            if mass > self.mass_thr and x_dig is None:
                x_dig = (jdata['x'] - self.x_min) / (self.x_max - self.x_min)
            if mass > self.mass:
                self.mass = mass
            in_target = np.zeros_like(target)
            t_step_start = time()
            while not np.all(in_target):
                t_step_delta = time() - t_step_start
                if jdata is not None:
                    current = jdata['x']
                    dist_to_x = np.abs(np.array(current) - target)
                    for i in range(self.action_dim):
                        if dist_to_x[i] < x_thr:
                            in_target[i] = 1
                if t_step_delta > self.t_max:
                    break
        t_cycle_delta = time() - t_cycle_start
        t_elapsed = t_cycle_delta / self.t_max
        state = self._construct_state(x_dig)
        self.step_count += 1
        sleep(self.t_max)
        jdata = self._post_target()
        dumped = jdata['d'] / self.m_max
        reward = self._calculate_reward(x_dig, dumped, n_collisions, t_elapsed)
        if self.debug:
            print(self.step_count, state, reward, self.dig_target)
        if self.step_count == self.step_count_max:
            print('Solver {0} will restart as it has reached maximum step count.'.format(self.env_id))
            done = True
        elif n_collisions > 0:
            #print('Solver {0} should restart due to unexpected collision!'.format(self.env_id))
            done = False # True
        else:
            done = False
        return state, reward, done, {'r': reward, 'l': self.step_count}

    def reset(self, delay=1.0, x_ind=4):

        # restart backend if there is already one

        current_mode = self._get_mode()
        self._get_mode(new_mode='RESTART')
        self._reset_id()
        self._start_backend()
        self._request_targets()
        self.target_idx = 0
        self._get_mode(new_mode=current_mode)

        # wait for the backend to start

        ready = False
        while not ready:
            jdata = self._post_target()
            if jdata is not None:
                if jdata['backend_running'] and jdata['x'] is not None and jdata['l'] is not None and jdata['m'] is not None:
                    ready = True
                else:
                    sleep(delay)
        state = self._construct_state(self.dig_target)
        self.step_count = 0
        return state

    def render(self, mode='human', close=False):
        pass

    def _get_mode(self, new_mode=None, uri='mode'):
        url = '{0}/{1}'.format(self.http_url, uri)
        if new_mode is not None:
            jdata = requests.post(url, json={'id': self.env_id, 'mode': new_mode}).json()
        else:
            jdata = requests.get(url, json={'id': self.env_id}).json()
        return jdata['mode']

    def _reset_id(self, uri='id'):
        url = '{0}/{1}'.format(self.http_url, uri)
        assigned = self.backend_assigned
        while assigned:
            try:
                j = requests.post(url, json={'id': self.env_id}).json()
                assigned = j['assigned']
            except Exception as e:
                print('Exception when reseting id:')
                print(e)
        self.backend_assigned = assigned

    def _start_backend(self, n_attempts=30, autostart=True, uri='id'):
        ready = False
        url = '{0}/{1}'.format(self.http_url, uri)
        print('Trying to start solver {0}...'.format(self.env_id))
        while not ready:
            assigned = self.backend_assigned
            attempt = 0
            if autostart:
                sleep(self.env_id)
                self.solver_proc = subprocess.Popen(self.solver_args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            while not assigned:
                try:
                    j = requests.get(url, json={'id': self.env_id}).json()
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
                self.backend_assigned = assigned
                print('Solver {0} has successfully started!'.format(self.env_id))
            else:
                print('Could not start solver {0} :( Trying again...'.format(self.env_id))

    def _request_targets(self, uri='targets', eps=1e-10):
        url = '{0}/{1}'.format(self.http_url, uri)
        target_list = []
        while len(target_list) <= 1:
            try:
                j = requests.get(url, json={'id': self.env_id}).json()
                target_list = j['targets']
            except Exception as e:
                print('Exception when requesting target list:')
                print(e)
        self.dig_target = (np.array(target_list[0]) - self.x_min) / (self.x_max - self.x_min + eps)
        self.emp_target = (np.array(target_list[1]) - self.x_min) / (self.x_max - self.x_min + eps)

    def _post_target(self, target=None, uri='p_target'):
        url = '{0}/{1}'.format(self.http_url, uri)
        if target is not None:
            target = target.tolist()
        try:
            jdata = requests.post(url, json={'id': self.env_id, 'y': target}).json()
        except Exception as e:
            print('Exception when posting new p_target:')
            print(e)
            jdata = None
        return jdata

    def _construct_state(self, x_dig, eps=1e-10):
        t_std = self.dig_target
        m_std = self.mass / (self.m_max + eps)
        state = np.hstack([x_dig, t_std, m_std])
        return state

    def _calculate_reward(self, x_dig, m_dumped, n_collisions, t_elapsed):
        dist_to_target = np.linalg.norm(x_dig - self.dig_target)
        r = self.dist_coeff * (1 - dist_to_target) + self.time_coeff * (1 - t_elapsed) + self.mass_coeff * m_dumped - self.collision_coeff * n_collisions
        return r