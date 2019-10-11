import numpy as np
import requests
from time import time
from collections import deque

# Data names

data_names = {
    'timestamps': ['Time'],
    'controls': [
        'Input_Slew RealValue',
        'Input_BoomLift RealValue',
        'Input_DipperArm RealValue',
        'Input_Bucket RealValue'
    ],
    'measurements': [
        'ForceR_Slew r',
        'Cylinder_BoomLift_L x',
        'Cylinder_DipperArm x',
        'Cylinder_Bucket x'
    ],
    'scores': [
        'massSensorBucketTeeth Volume'
    ]
}

# Component names

components = [p.split(' ')[0] for p in data_names['measurements']]
parameters = [p.split(' ')[1] for p in data_names['measurements']]
score_components = [p.split(' ')[0] for p in data_names['scores']]
score_parameters = [p.split(' ')[1] for p in data_names['scores']]
component_controls = [p.split(' ')[0] for p in data_names['controls']]

# thresholds

move_thr = 3
time_thr = 0.01

# Dimensions

pid_dim = 3
time_dim = 4
trajectory_dim = 5
action_dim = 4

# Other parameters

n_attempts_to_reach_target = 64
n_iterations_stay = 1
pid_gain_limits = np.vstack([
    100 * np.ones(action_dim),
    1 * np.ones(action_dim),
    10 * np.ones(action_dim)
])
# RL agent

agent = '127.0.0.1:5000'
status_uri = 'mode'
trajectory_uri = 'trajectory'
control_uri = 'controls'

# User input file

user_input_fname = 'C:\\Users\\iotli\\\PycharmProjects\\SmartExcavator\\user_input\\user_input.txt'
open(user_input_fname, 'w').close()

def pid_controls(points_list, target, delta_time, integ_prev, gains):
    points = np.array(points_list)
    p = target - points[-1, :]
    error_prev = target - points[-2, :]
    d = (p - error_prev) / delta_time
    i = integ_prev + (p + error_prev) / 2 * delta_time
    controls = - (gains[0, :] * p + gains[1, :] * i + gains[2, :] * d)
    return controls, i

def get_status():
    uri = 'http://{0}/{1}'.format(agent, status_uri)
    try:
        r = requests.get(uri)
        jdata = r.json()
        status = jdata['mode']
    except Exception as e:
        print(e)
        status = None
    return status

def deque_to_list(points_deque):
    points_list = []
    for p in points_deque:
        points_list.append(p)
    return points_list

def get_trajectory(points, score):
    uri = 'http://{0}/{1}'.format(agent, trajectory_uri)
    jdata = {'x': points[-1], 'score': score}
    try:
        r = requests.get(uri, json=jdata)
        jdata = r.json()
        trajectory = np.array(jdata['y'])
        timestamps = np.array(jdata['t'])
    except Exception as e:
        print(e)
        trajectory = None
        timestamps = None
    return trajectory, timestamps

def points_to_deltas(points, target_point):
    deltas = []
    for point in points:
        deltas.append([])
        for p, t in zip(point, target_point):
            deltas[-1].append(np.abs(p - t))
    return deltas

def get_pid_gains(deltas, delta_start, delta_end, in_target, time_passed, time_limit, done):
    uri = 'http://{0}/{1}'.format(agent, control_uri)
    jdata = {'deltas': deltas, 'delta_start': delta_start, 'delta_end': delta_end, 'in_target': in_target, 'time': time_passed, 'time_limit': time_limit, 'done': done}
    try:
        r = requests.get(uri, json=jdata)
        jdata = r.json()
        controls = jdata['controls']
        gains = pid_gain_limits * np.array(controls)
        success = True
    except Exception as e:
        print(e)
        gains = np.vstack([[0 for _ in range(action_dim)] for _ in range(pid_dim)])
        success = False
    print(gains)
    return gains, success

def generate_reset_trajectory(trajectory, trajectory_idx):
    reset_trajectory = []
    for i in np.arange(trajectory_idx - 1, 0, -1):
        reset_trajectory.append(trajectory[i, :])
    reset_trajectory.append(trajectory[-1, :])
    return np.array(reset_trajectory)

def reset(current_value, target_point):
    error = target_point - current_value
    controls_ = - error
    dist_r = np.sqrt(np.sum(error[0] ** 2))
    dist_x = np.sqrt(np.sum(error[1:] ** 2))
    if dist_x <= move_thr:
        if dist_r <= move_thr:
            start_point_reached = True
            controls = np.zeros(action_dim)
        else:
            start_point_reached = False
            controls = np.hstack([controls_[0], np.zeros(3)])
    else:
        start_point_reached = False
        controls = np.hstack([0, controls_[1:]])
    return controls, start_point_reached


##### I N I T  S C R I P T #####


def initScript():

    # components to get real values

    GObject.data['component_objects'] = [
        GSolver.getParameter(components[index], parameters[index]) for index in range(len(components))
    ]
    GObject.data['score_objects'] = [
        GSolver.getParameter(score_components[index], score_parameters[index]) for index in range(len(score_components))
    ]

    # other variables

    GObject.data['mode'] = 'USER'
    GObject.data['is_trajectory_set'] = False
    GObject.data['trajectory'] = []
    GObject.data['timestamps'] = []
    GObject.data['trajectory_idx'] = 0
    GObject.data['in_target'] = np.zeros(action_dim)
    GObject.data['start_time'] = time()
    GObject.data['last_time'] = time()
    GObject.data['ticks_since_last_time'] = 0.0
    GObject.data['score'] = 0
    GObject.data['done'] = False
    GObject.data['n_attempts'] = 0
    GObject.data['stuck'] = False
    GObject.data['are_pid_gains_set'] = False
    GObject.data['target_set_time'] = time()

    # initial state

    GObject.data['points'] = deque(maxlen=time_dim)
    real_values = [x.value() for x in GObject.data['component_objects']]
    for _ in range(time_dim):
        GObject.data['points'].append(real_values)
    GObject.data['delta_start'] = [0 for _ in range(action_dim)]
    GObject.data['delta_end'] = [0 for _ in range(action_dim)]
    GObject.data['controls'] = np.zeros(action_dim)
    GObject.data['integ_prev'] = np.zeros(action_dim)
    GObject.data['pid_gains'] = np.zeros((pid_dim, action_dim))


##### C A L L  S C R I P T #####


def callScript(deltaTime, simulationTime):

    # check if some time passed

    time_passed = False
    t_now = time()
    t_last = GObject.data['last_time']
    GObject.data['ticks_since_last_time'] += 1.0
    if t_now - t_last > time_thr:
        time_passed = True

    # get current position and score

    real_values = [x.value() for x in GObject.data['component_objects']]
    score_values = [x.value() for x in GObject.data['score_objects']]

    # if the simulator is under user control write position to a file and check mode every iteration

    if GObject.data['mode'] == 'USER':
        t = time() - GObject.data['start_time']
        line = ','.join([str(t)] + [str(val) for val in real_values])
        with open(user_input_fname, 'a') as f:
            f.write(line + '\n')
        mode = get_status()
        if mode is not None:
            GObject.data['mode'] = mode

    # update state only if some time passed

    if time_passed:
        points = GObject.data['points']
        points.append(real_values)
        GObject.data['points'] = points

    # default control and time values

    change_controls = False
    controls = np.zeros(action_dim)

    # if the simulator is under AI control

    if GObject.data['mode'].startswith('AI'):

        # if the trajectory is not set, ask for it immediately

        if GObject.data['is_trajectory_set'] == False:

            # request trajectory

            points = GObject.data['points']
            points_list = deque_to_list(points)
            trajectory, timestamps = get_trajectory(points_list, score_values[0])

            # in case of success

            if trajectory is not None and timestamps is not None:
                GObject.data['timestamps'] = timestamps
                GObject.data['trajectory'] = trajectory
                GObject.data['trajectory_idx'] = 0
                GObject.data['is_trajectory_set'] = True
                GObject.data['ticks_since_last_time'] = 0.0
                GObject.data['n_attempts'] = 0

                # PID control gains for the 1-st target in the trajectory

                points = GObject.data['points']
                idx = GObject.data['trajectory_idx']
                target = trajectory[idx, :]
                deltas = points_to_deltas(points, target)
                last_delta = deltas[-1]
                if np.all(np.array(last_delta) <= move_thr):
                    in_target = True
                else:
                    in_target = False
                gains, success = get_pid_gains(
                    deltas,
                    GObject.data['delta_start'],
                    GObject.data['delta_end'],
                    in_target,
                    time() - GObject.data['target_set_time'],
                    time() - GObject.data['target_set_time'],
                    GObject.data['done']
                )
                GObject.data['delta_start'] = last_delta
                GObject.data['target_set_time'] = time()
                GObject.data['pid_gains'] = gains
                GObject.data['are_pid_gains_set'] = success
                GObject.data['integ_prev'] = np.zeros(action_dim)
                GObject.data['done'] = False

        # move

        if GObject.data['is_trajectory_set'] == True and time_passed:

            # calculate deltas and time required

            trajectory = GObject.data['trajectory']
            timestamps = GObject.data['timestamps']
            idx = GObject.data['trajectory_idx']
            target = trajectory[idx, :]
            deltas = points_to_deltas(points, target)
            time_limit = timestamps[idx]

            # check if we reach the target point

            target_reached = np.zeros(action_dim)
            for i in range(action_dim):
                if deltas[-1][i] < move_thr:
                    target_reached[i] = 1
                else:
                    target_reached[i] = 0
            for i in range(action_dim):
                if target_reached[i] == 1:
                    GObject.data['in_target'][i] += 1
                elif target_reached[i] == 0:
                    GObject.data['in_target'][i] = 0

            # if the target point has been reached

            if np.all(GObject.data['in_target'] >= n_iterations_stay):

                # nulify in_target and attempt count

                GObject.data['in_target'] = np.zeros(action_dim)
                GObject.data['n_attempts'] = 0

                # if the trajectory has been completed

                if GObject.data['trajectory_idx'] >= len(GObject.data['trajectory']) - 1:

                    # remember done and last delta

                    GObject.data['done'] = True
                    GObject.data['is_trajectory_set'] = False
                    GObject.data['delta_end'] = list(deltas[-1])

                    # check status

                    mode = get_status()
                    if mode is not None:
                        GObject.data['mode'] = mode

                        # if mode is USER, clean previous user input

                        if mode == 'USER':
                            open(user_input_fname, 'w').close()
                            print('\nUSER INPUT\n')

                # if there are still points in the trajectory or time limit has been reached, just continue

                else:

                    # remember start and end delta

                    delta_start = GObject.data['delta_start']
                    delta_end = list(deltas[-1])

                    # increment trajectory index

                    GObject.data['trajectory_idx'] += 1

                    # switch the target

                    trajectory = GObject.data['trajectory']
                    idx = GObject.data['trajectory_idx']
                    target = trajectory[idx, :]

                    # recalculate deltas and time spent

                    deltas = points_to_deltas(points, target)
                    last_delta = deltas[-1]
                    time_spent = time() - GObject.data['target_set_time']

                    # request PID gains

                    done = True # False
                    if np.all(np.array(last_delta) <= move_thr):
                        in_target = True
                    else:
                        in_target = False
                    gains, success = get_pid_gains(
                        deltas,
                        delta_start,
                        delta_end,
                        in_target,
                        time_spent,
                        time_limit,
                        done
                    )
                    GObject.data['delta_start'] = list(last_delta)
                    GObject.data['pid_gains'] = gains
                    GObject.data['are_pid_gains_set'] = success
                    GObject.data['integ_prev'] = np.zeros(action_dim)
                    GObject.data['target_set_time'] = time()
                    GObject.data['done'] = False

            # if time limit has been reached, we request new PID gains without switching the target

            elif time() - GObject.data['target_set_time'] > time_limit:

                # increment attempt count

                GObject.data['n_attempts'] += 1

                # remember start and end delta for mover score calculation

                delta_start = GObject.data['delta_start']
                delta_end = list(deltas[-1])

                # request new PID control gains

                last_delta = deltas[-1]
                time_spent = time() - GObject.data['target_set_time']
                if GObject.data['n_attempts'] >= n_attempts_to_reach_target:
                    done = True
                    GObject.data['n_attempts'] = 0
                else:
                    done = False
                if np.all(np.array(last_delta) <= move_thr):
                    in_target = True
                else:
                    in_target = False
                gains, success = get_pid_gains(
                    deltas,
                    delta_start,
                    delta_end,
                    in_target,
                    time_spent,
                    time_limit,
                    done
                )

                # we update delta_start only in case of no attempts left

                if done:
                    GObject.data['delta_start'] = list(last_delta)

                GObject.data['pid_gains'] = gains
                GObject.data['are_pid_gains_set'] = success
                GObject.data['integ_prev'] = np.zeros(action_dim)
                GObject.data['target_set_time'] = time()
                GObject.data['done'] = False

            # recalculate PID controls

            points = GObject.data['points']
            points_list = deque_to_list(points)
            delta_time = time() - GObject.data['last_time']
            integ_prev = GObject.data['integ_prev']
            controls, integ_prev = pid_controls(points_list, target, delta_time, integ_prev, GObject.data['pid_gains'])
            GObject.data['integ_prev'] = integ_prev
            change_controls = True

    # change control

    if time_passed:

        if change_controls:

            # set input values

            GObject.data['controls'] = controls
            cv = GObject.data['controls']
            cc = component_controls
            for i in range(action_dim):
                GDict[cc[i]].setInputValue(cv[i])

        # nulify time

        GObject.data['last_time'] = time()
        GObject.data['ticks_since_last_time'] = 0