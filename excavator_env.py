import winreg, subprocess, gym, requests
import numpy as np
import tensorflow as tf
from time import sleep, time
from learn_demonstration_policy import create_model

class ExcavatorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, id,
        mws='C:\\Users\\iotli\\PycharmProjects\\SmartExcavator\\mws\\env.mws',
        obs_dim=9+4,
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
        self.n_steps = 14
        self.n_series = 4
        self.step_count_max = self.n_series * self.n_steps
        self.env_id = id
        self.backend_assigned = False
        self.target_list = []
        self.target_idx = 0
        self.http_url = http_url

        # state, action and reward coefficients

        self.obs_idx = 0
        self.obs_stack = np.zeros((self.n_steps, self.obs_dim - self.action_dim))
        self.stick_to_demonstration_policy = 0.95
        self.time_coeff = 0.5
        self.dist_coeff = 0.5
        self.mass_coeff = 1.0
        self.collision_coeff = 10.0
        self.dist_thr = 0.01
        self.mass_thr = 0.05
        self.collision_thr = 0
        self.x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
        self.x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
        self.m_max = 1000
        self.t_max = 3.0
        self.last_state = np.zeros(obs_dim - action_dim)
        self.last_demo_action = np.zeros(action_dim)

        # demonstration policy

        self.model = self._load_demo_policy()

        # observation and action spaces

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float)
        self.discrete_action = discrete_action
        if self.discrete_action is None:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float)
        else:
            self.action_space = gym.spaces.Discrete(self.discrete_action * action_dim)

    def step(self, action, x_ind=4, m_ind=-1, x_thr=5.0):
        real_action = self.stick_to_demonstration_policy * self.last_demo_action + (1 - self.stick_to_demonstration_policy) * action
        target = np.clip(self.x_min + (self.x_max - self.x_min) * real_action, self.x_min, self.x_max)
        jdata = self._post_target(target)
        in_target = np.zeros_like(target)
        while not np.all(in_target):
            t_delta = time() - self.last_step_time
            if jdata is not None:
                current = jdata['x']
                dist_to_x = np.abs(np.array(current) - target)
                for i in range(self.action_dim):
                    if dist_to_x[i] < x_thr:
                        in_target[i] = 1
            if t_delta > self.t_max:
                break
            jdata = self._post_target()
        state = self._construct_state(jdata)
        demo_action = self._predict_demo_action(state)
        x = state[:x_ind]
        m = state[m_ind]
        c = jdata['c']
        self.step_count += 1
        reward, switch_target, restart_required = self._calculate_reward(x, m, c, t_delta)
        print(self.obs_idx, target, reward, switch_target)
        if self.step_count >= self.step_count_max - 1:
            print('Solver {0} will restart as it has reached maximum step count.'.format(self.env_id))
            done = True
        elif restart_required:
            print('Solver {0} will restart due to unexpected collision!'.format(self.env_id))
            self.obs_stack = np.zeros((self.n_steps, self.obs_dim - self.action_dim))
            self.obs_idx = 0
            done = True
        elif switch_target:
            if self.target_idx >= len(self.target_list) - 1:
                self.target_idx = 0
            else:
                self.target_idx += 1
            done = False
        else:
            done = False
        self.last_state = state.copy()
        self.last_demo_action = demo_action.copy()
        self.last_step_time = time()
        return np.hstack([state, demo_action]), reward, done, {'r': reward, 'l': self.step_count}

    def reset(self, delay=1.0, x_ind=4):

        # restart backend if there is already one

        current_mode = self._get_mode()
        self._get_mode(new_mode='RESTART')
        self._reset_id()
        self._start_backend()
        self._request_target_list()
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
        state = self._construct_state(jdata)
        demo_action = self._predict_demo_action(state)
        self.last_step_time = time()
        self.step_count = 0
        self.last_state = state.copy()
        self.last_demo_action = demo_action.copy()
        return np.hstack([state, demo_action])

    def render(self, mode='human', close=False):
        pass

    def _load_demo_policy(self, checkpoint_path='policies/demonstration/last.ckpt'):
        model = create_model(self.obs_dim - self.action_dim, self.action_dim, series_len=self.n_steps)
        model.load_weights(checkpoint_path)
        return model

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
        state = np.hstack([target - l_std, target - x_std, m_std])
        return state

    def _predict_demo_action(self, state):
        self.obs_stack[self.obs_idx, :] = state
        state_reshaped = self.obs_stack.reshape(1, self.n_steps, self.obs_dim - self.action_dim)
        action = self.model.predict(state_reshaped)[0]
        self.obs_idx += 1
        return action

    def _calculate_reward(self, x, m, c, t):
        target_bit = self.target_idx % 2
        switch_target = False
        restart_required = False
        dist_to_target = np.linalg.norm(x - self.target_list[self.target_idx])
        time_elapsed = t / self.t_max
        r = self.dist_coeff * (1 - dist_to_target) + self.time_coeff * (1 - time_elapsed)  - self.collision_coeff * c
        if self.obs_idx == self.n_steps:
            switch_target = True
            self.obs_stack = np.zeros((self.n_steps, self.obs_dim - self.action_dim))
            self.obs_idx = 0
        elif target_bit == 0:
            if dist_to_target <= self.dist_thr:
                if m > self.mass_thr:
                    switch_target = True
        else:
            r += self.mass_coeff * m
            if dist_to_target <= self.dist_thr:
                if m < self.mass_thr:
                    switch_target = True
                    self.obs_stack = np.zeros((self.n_steps, self.obs_dim - self.action_dim))
                    self.obs_idx = 0
        if c > self.collision_thr:
            restart_required = True
        return r, switch_target, restart_required
