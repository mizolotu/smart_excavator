import winreg, subprocess, gym, requests
import numpy as np
from time import sleep, time

class ExcavatorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, id,
        mws='C:\\Users\\iotli\\PycharmProjects\\SmartExcavator\\mws\\env.mws',
        state_dim=10,
        action_dim=4,
        http_url='http://127.0.0.1:5000',
        discrete_action=None,
    ):
        super(ExcavatorEnv, self).__init__()
        regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
        (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
        solverpath += r'\Bin\MeveaSolver.exe'
        winreg.CloseKey(regkey)
        self.solver_args = [solverpath, r'/loadmws', mws, r'/saveplots', r'/silent']
        self.solver_proc = None
        self.episode_count = 0
        self.step_count = 0
        self.step_count_max = 100
        self.env_id = id
        self.backend_assigned = False
        self.target_list = []
        self.target_idx = 0
        self.http_url = http_url
        self.dist_coeff = 0.5
        self.orientation_coeff = 0.2
        self.mass_coeff = 0.8
        self.collision_coeff = 10.0
        self.dist_thr = 0.01
        self.orientation_thr = 0.05
        self.mass_thr = 0.05
        self.collision_thr = 0
        self.x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
        self.x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
        self.t_max = np.array([16.38471214911517, 7.04791988497195, 4.854493033885956, 3.783558408419291])
        self.m_max = 1000
        self.last_x = np.zeros(action_dim)
        self.last_step_time = time()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float)
        self.discrete_action = discrete_action
        if self.discrete_action is None:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float)
        else:
            self.action_space = gym.spaces.Discrete(self.discrete_action * action_dim)

    def step(self, action, x_ind=4, m_ind=-2):
        target = self._action_target(action)
        jdata_before_action = self._post_target(target)
        delay = self._calculate_delay(jdata_before_action['x'], target[:x_ind])
        sleep(np.maximum(0, delay - time() + self.last_step_time))
        jdata = self._post_target()
        state = self._construct_state(jdata)
        x = state[:x_ind] # first 4 features
        m = state[m_ind] # second last feature
        c = jdata['c']
        self.step_count += 1
        reward, switch_target, restart_required = self._calculate_reward(x, m, c)
        if self.step_count >= self.step_count_max:
            print('Solver {0} will restart as it has reached maximum step count.'.format(self.env_id))
            done = True
        elif restart_required:
            print('Solver {0} will restart due to unexpected collision!'.format(self.env_id))
            done = True
        elif switch_target:
            if self.target_idx >= len(self.target_list) - 1:
                self.target_idx = 0
            else:
                self.target_idx += 1
            done = False
        else:
            done = False
        self.last_x = state[:x_ind]
        self.last_step_time = time()
        return state, reward, done, {'r': reward, 'l': self.step_count}

    def reset(self, delay=1.0, x_ind=4):

        # restart backend if there is already one

        current_mode = self._get_mode()
        self._get_mode(new_mode='RESTART')
        self._reset_id()
        self._start_backend()
        self._request_target_list()
        self.target_idx = 0
        self._get_mode(current_mode)

        # wait for the backend to start

        ready = False
        while not ready:
            jdata = self._post_target()
            if jdata is not None:
                if jdata['backend_running'] and jdata['x'] is not None and jdata['l'] is not None and jdata['m'] is not None:
                    ready = True
                else:
                    sleep(delay)
        state = self._construct_state(jdata)
        self.last_step_time = time()
        self.step_count = 0
        self.last_x = state[:x_ind]
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

    def _request_target_list(self, uri='targets', eps=1e-10):
        url = '{0}/{1}'.format(self.http_url, uri)
        target_list = []
        while len(target_list) == 0:
            try:
                j = requests.get(url, json={'id': self.env_id}).json()
                target_list = j['targets']
            except Exception as e:
                print('Exception when requesting target list:')
                print(e)
        self.target_list = []
        for t in target_list:
            x = np.array(t)
            self.target_list.append((x - self.x_min) / (self.x_max - self.x_min + eps))

    def _action_target(self, action):
        if self.discrete_action is None:
            action = np.array(action)
            action_std = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
            action_state_deltas = np.abs(action_std - self.last_x)
            action_delays = action_state_deltas * self.t_max
            action_coeff = action_delays / np.max(action_delays)
            #action_target_idx = action_state_deltas.argmax()
            #action_none = np.array([np.nan for _ in action])
            #action_none[action_target_idx] = action_std[action_target_idx]
            target = np.hstack([self.x_min + action_std * (self.x_max - self.x_min), action_coeff])
        else:
            action_target_idx = action // self.discrete_action
            action_target_amp = (action % self.discrete_action) / (self.discrete_action - 1)
            action_std = np.array([np.nan for _ in range(self.action_space.n // self.discrete_action)])
            action_std[action_target_idx] = action_target_amp
            target = self.x_min + action_std * (self.x_max - self.x_min)
        return target

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

    def _calculate_delay(self, x, y, default_delay=0.1, eps=1e-10):
        delay = default_delay
        if x is not None:
            delta = y - x
            idx = np.ones(len(x))
            idx[np.isnan(delta)] = 0
            idx_ = idx.argmax()
            delay = self.t_max[idx_] * np.abs(delta[idx_]) / (self.x_max[idx_] - self.x_min[idx_] + eps)
        return delay

    def _construct_state(self, jdata, eps=1e-10):
        x = np.array(jdata['x'])
        l = np.array(jdata['l'])
        m = jdata['m']
        x_std = (x - self.x_min) / (self.x_max - self.x_min + eps)
        l_std = (l - self.x_min) / (self.x_max - self.x_min + eps)
        m_std = m / (self.m_max + eps)
        target = self.target_list[self.target_idx]
        target_bit = self.target_idx % 2
        state = np.hstack([x_std, target, m_std, target_bit]) # current position + target position + ground mass in bucket + target type
        return state

    def _calculate_reward(self, x, m, c):
        target_bit = self.target_idx % 2
        switch_target = False
        restart_required = False
        dist_to_target = np.linalg.norm(x[:-1] - self.target_list[self.target_idx][:-1])
        r = 1 - self.dist_coeff * dist_to_target
        if target_bit == 0:
            if dist_to_target <= self.dist_thr:
                dist_to_orientation = np.abs(self.observation_space.high[0] - x[-1])
                r += 1 - dist_to_orientation * self.orientation_coeff + m * self.mass_coeff
                if dist_to_orientation < self.orientation_thr and m > self.mass_thr:
                    switch_target = True
            else:
                r -= c * self.collision_coeff
                if c > self.collision_thr:
                    restart_required = True
        else:
            if dist_to_target <= self.dist_thr:
                dist_to_orientation = np.abs(self.observation_space.low[0] - x[-1])
                r += (1 - dist_to_orientation) * self.orientation_coeff
                if dist_to_orientation < self.orientation_thr and m < self.mass_thr:
                    switch_target = True
            else:
                r -= c * self.collision_coeff
                if c > self.collision_thr:
                    restart_required = True
        return r, switch_target, restart_required
