import winreg, subprocess, gym, requests, psutil
import numpy as np

from time import sleep, time

class ExcavatorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, id, policy,
        obs_dim=5,
        action_dim=4,
        http_url='http://127.0.0.1:5000'
    ):

        super(ExcavatorEnv, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # id and counts

        self.env_id = id
        self.policy = policy
        self.episode_count = 0
        self.step_count = 0
        self.n_steps = 8
        self.step_count_max = 8
        self.backend_assigned = False
        self.dig_target = None
        self.emp_target = None
        self.emp_trajectory = None
        self.http_url = http_url

        # thresholds and coefficients

        self.turn_coeff = 1.0
        self.move_coeff = 1.0
        self.mass_coeff = 1.0
        self.mass_thr = 0.05
        self.x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
        self.x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
        self.d_min = np.array([-0.1, -0.3, -0.5, -0.8])
        self.d_max = np.array([0.1, 0.3, 0.5, 0.8])
        self.m_max = 1000.0
        self.t_max = 60.0
        self.delay_max = 3.0

        # collision accounting

        self.collision_coeff = 0.0
        self.restart_on_collision = False

        # observation and action spaces

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim * self.n_steps,), dtype=np.float)
        self.debug = False

        # other

        self.dist_d_max_d_min = np.linalg.norm(self.d_max[1:] - self.d_min[1:])

    def step(self, action, a_thr=3.0, x_thr=5.0):

        # current position

        jdata = self._post_target()
        x_current = (jdata['x'] - self.x_min) / (self.x_max - self.x_min)

        # generate full trajectory

        action = np.clip(action, self.action_space.low, self.action_space.high)  # action values are now in range -1..1
        action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)  # action values are now in range 0..1
        dig_deltas_std = action.reshape(self.n_steps, self.action_dim)
        dig_deltas = np.vstack([self.d_min + x * (self.d_max - self.d_min) for x in dig_deltas_std])  # action values are now in range d_min..d_max
        if self.policy == 'residual':
            dig_the_target = np.vstack([x + y for x, y in zip(dig_deltas, self.dig_trajectory)])  # 1..n_steps
        else:
            dig_the_target = np.vstack([x + self.dig_target for x in dig_deltas])  # 1..n_steps

        # calculate diging distance

        d_dig = 0
        for i in range(1, self.n_steps):
            d_dig += np.linalg.norm(dig_the_target[i, 1:] - dig_the_target[i-1, 1:])
        d_dig /= (self.n_steps - 1)
        d_dig /= self.dist_d_max_d_min

        # complete the trajectory

        turn_to_the_target = np.hstack([dig_the_target[0, 0], x_current[1:]])  # 0
        prepare_for_the_turn_to_the_dumper = np.hstack([dig_the_target[-1, 0], self.emp_trajectory[0, 1:]])  # n_steps + 1
        trajectory = np.vstack([turn_to_the_target, dig_the_target, prepare_for_the_turn_to_the_dumper, self.emp_trajectory])  # n_steps + 2..n_steps + 2 + emp_trajectory_len

        # move along the trajectory

        x_dig = None
        n_collisions = 0
        mass_max = 0
        soil_grabbed = 0
        t_start = time()
        for i in range(trajectory.shape[0]):
            target = np.clip(self.x_min + (self.x_max - self.x_min) * trajectory[i, :], self.x_min, self.x_max)
            jdata = self._post_target(target)
            n_collisions += jdata['c']
            mass = jdata['m'] / self.m_max
            if i <= self.n_steps and mass > self.mass_thr and x_dig is None:
                x_dig = (jdata['x'] - self.x_min) / (self.x_max - self.x_min)
            if mass > mass_max:
                mass_max = mass
            in_target = np.zeros_like(target)
            in_angle_target = 0
            t_step_start = time()
            while not np.all(in_target):
                jdata = self._post_target()
                t_step_delta = time() - t_step_start
                if jdata is not None:
                    current = jdata['x']
                    dist_to_x = np.abs(np.array(current) - target)
                    if np.abs(current[0] - target[0]) < a_thr:
                        in_angle_target = 1
                    for j in range(self.action_dim):
                        if dist_to_x[j] < x_thr:
                            in_target[j] = 1
                if i in [0, self.n_steps + 2] and in_angle_target:
                    break
                elif i not in [0, self.n_steps + 2] and t_step_delta > self.delay_max:
                    break
            if i == self.n_steps + 2:
                soil_grabbed = mass
        state = self._construct_state(x_dig, mass_max)
        self.step_count += 1
        reward = self._calculate_reward(x_dig, d_dig, soil_grabbed, n_collisions)
        if self.debug:
            print(self.step_count, reward, time() - t_start)
            print(self.env_id, self.step_count, state, reward, x_dig)
        if self.step_count == self.step_count_max:
            print('After working for {0} seconds, solver {1} stops as it has reached the maximum step count.'.format(time() - self.time_start, self.env_id))
            done = True
        elif n_collisions > 0 and self.restart_on_collision:
            print('Solver {0} will restart due to unexpected collision!'.format(self.env_id))
            done = True
        else:
            done = False
        return state, reward, done, {'r': reward, 'm': soil_grabbed, 'd': d_dig}

    def reset(self, delay=1.0):

        self._request_targets()
        self._generate_dig_trajectory()
        self._generate_emp_trajectory()
        self.target_idx = 0

        # wait for the backend to start

        ready = False
        while not ready:
            jdata = self._post_target()
            if jdata is not None:
                if jdata['backend_running'] and jdata['x'] is not None and jdata['l'] is not None and jdata['m'] is not None:
                    ready = True
                else:
                    sleep(delay)

        state = self._construct_state()
        self.step_count = 0
        self.time_start = time()
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

    def _request_targets(self, uri='targets', eps=1e-10):
        url = '{0}/{1}'.format(self.http_url, uri)
        target_list = [None, None]
        while None in target_list:
            try:
                j = requests.get(url, json={'id': self.env_id}).json()
                target_list = j['targets']
            except Exception as e:
                print('Exception when requesting target list:')
                print(e)
        self.dig_target = (np.array(target_list[0]) - self.x_min) / (self.x_max - self.x_min + eps)
        self.emp_target = (np.array(target_list[1]) - self.x_min) / (self.x_max - self.x_min + eps)

    def _generate_dig_trajectory(self, fname='data/dig.txt'):
        dig = []
        with open(fname) as f:
            lines = f.readlines()
        for line in lines:
            dig.append([float(x) for x in line.split(',')])
        dig = np.vstack(dig)
        dig_old_angle = np.mean(dig[:, 0])
        dig_new_angles = dig[:, 0] - dig_old_angle + self.dig_target[0]
        self.dig_trajectory = dig
        self.dig_trajectory[:, 0] = dig_new_angles

    def _generate_emp_trajectory(self, fname='data/emp.txt'):
        emp = []
        with open(fname) as f:
            lines = f.readlines()
        for line in lines:
            emp.append([float(x) for x in line.split(',')])
        emp = np.vstack(emp)
        emp_old_angle = np.mean(emp[:, 0])
        emp_new_angles = emp[:, 0] - emp_old_angle + self.emp_target[0]
        self.emp_trajectory = emp
        self.emp_trajectory[:, 0] = emp_new_angles

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

    def _construct_state(self, x_dig=None, soil_taken=None):
        state = np.zeros(self.obs_dim)
        if x_dig is not None:
            state[:self.action_dim] = x_dig - self.dig_target
        if soil_taken is not None:
            state[self.action_dim] = soil_taken
        return state

    def _calculate_reward(self, x_dig, d_dig, m_grabbed, n_collisions):
        if x_dig is None or m_grabbed == 0:
            r = -1
        else:
            dist_to_target = np.abs(x_dig[0] - self.dig_target[0]) / (self.d_max[0] - self.d_min[0])
            r = self.turn_coeff * (1 - dist_to_target) + self.move_coeff * (1 - d_dig) + self.mass_coeff * m_grabbed - self.collision_coeff * n_collisions
        return r