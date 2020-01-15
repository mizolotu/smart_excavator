import sys, requests, json
import numpy as np
from time import time, sleep

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
        'massSensorBucketTeeth Mass'
    ],
    'collisions': [
        'Ground nminor',
        'Ground nserious',
        'Ground ncritical',
        'Dumper nminor',
        'Dumper nserious',
        'Dumper ncritical',
    ]
}

# Component names

components = [p.split(' ')[0] for p in data_names['measurements']]
parameters = [p.split(' ')[1] for p in data_names['measurements']]
score_components = [p.split(' ')[0] for p in data_names['scores']]
score_parameters = [p.split(' ')[1] for p in data_names['scores']]
collision_components = [p.split(' ')[0] for p in data_names['collisions']]
collision_parameters = [p.split(' ')[1] for p in data_names['collisions']]
component_controls = [p.split(' ')[0] for p in data_names['controls']]

# gains and thresholds

p_controller_frequency = 1000 # Hz
eps = 1e-10 # to prevent dividing by zero
target_thr = 2.5

# dimensions

state_dim = 4
action_dim = 4

# http parameters

http_url = 'http://127.0.0.1:5000'
register_uri = 'register'
mode_uri = 'mode'
target_uri = 'p_target'

# User input file

user_input_fname = 'C:\\Users\\iotli\\\PycharmProjects\\SmartExcavator\\user_input\\user_input.txt'
open(user_input_fname, 'w').close()

def register():
    uri = '{0}/{1}'.format(http_url, register_uri)
    try:
        r = requests.get(uri)
        jdata = r.json()
        eid = jdata['id']
    except Exception as e:
        print(e)
        eid = None
    return eid

def pid_controls(current, previous, action, delta_time, integ_prev, gains=[[100, 100, 100, 100], [0.5, 0.01, 0.01, 0.05], [5, 0.5, 0.5, 0.5]]):
    previous = np.array(previous)
    current = np.array(current)
    target = np.array(action[:action_dim])
    if len(action) > action_dim:
        gains = np.array(action[action_dim:])
    else:
        gains = np.array(gains)
    p = target - current
    error_prev = target - previous
    d = (p - error_prev) / delta_time
    i = integ_prev + (p + error_prev) / 2 * delta_time
    gains = np.array(gains)
    controls = - (gains[0, :] * p + gains[1, :] * i + gains[2, :] * d)
    return controls, i

def p_controls(current, action, gains=[100, 100, 100, 100]):
    current = np.array(current)
    target = np.array(action[:action_dim])
    if len(action) > action_dim:
        gains = np.array(action[action_dim:])
    else:
        gains = np.array(gains)
    p = target - current
    controls = np.nan_to_num(-p * gains)
    return controls

def get_mode(id):
    uri = '{0}/{1}'.format(http_url, mode_uri)
    try:
        r = requests.get(uri, json={'id': id})
        jdata = r.json()
        mode = jdata['mode']
    except Exception as e:
        print(e)
        mode = None
    return mode

def get_target(id, point, last_point, time_passed, ground_mass, collisions=0):
    uri = '{0}/{1}'.format(http_url, target_uri)
    jdata = {'id': id, 'x': point, 'l': last_point, 't': time_passed, 'm': ground_mass, 'c': collisions}
    try:
        r = requests.get(uri, json=jdata)
        jdata = r.json()
        target = jdata['y']
        mode = jdata['mode']
    except Exception as e:
        print(e)
        target = None
        mode = None
    return target, mode


##### I N I T  S C R I P T #####


def initScript():

    # try to register

    env_id = None
    n_attempts = 0
    attempts_max = 10
    while env_id is None:
        env_id = register()
        sleep(1)
        n_attempts += 1
        if n_attempts >= attempts_max:
            sys.exit(1)
            pass
        print('Trying to register...')
    print('Successfully registerd with id {0}!'.format(env_id))
    GObject.data['id'] = env_id

    # components to get real values

    GObject.data['component_objects'] = [
        GSolver.getParameter(components[index], parameters[index]) for index in range(len(components))
    ]
    GObject.data['score_objects'] = [
        GSolver.getParameter(score_components[index], score_parameters[index]) for index in range(len(score_components))
    ]
    GObject.data['collision_objects'] = [
        GSolver.getParameter(collision_components[index], collision_parameters[index]) for index in range(len(collision_components))
    ]

    # initial state

    GObject.data['mode'] = 'USER'
    GObject.data['is_target_set'] = False
    GObject.data['in_target'] = [0 for _ in range(action_dim)]
    GObject.data['score'] = 0
    GObject.data['stuck'] = False
    GObject.data['start_time'] = time()
    GObject.data['target_set_time'] = time()
    GObject.data['last_time'] = time()
    GObject.data['previous_position'] = [x.value() for x in GObject.data['component_objects']]
    GObject.data['current_position'] = [x.value() for x in GObject.data['component_objects']]
    GObject.data['integ_prev'] = 0
    GObject.data['time_passed'] = 0
    GObject.data['target_position'] = None


##### C A L L  S C R I P T #####


def callScript(deltaTime, simulationTime):

    # check if required time passed

    time_passed = False
    t_now = time()
    t_last = GObject.data['last_time']
    t_delta = t_now - t_last
    if t_delta > 1.0 / p_controller_frequency:
        time_passed = True
    GObject.data['time_passed'] = t_delta

    # get current position and score

    real_values = [x.value() for x in GObject.data['component_objects']]
    score_values = [x.value() for x in GObject.data['score_objects']]
    collision_values = [x.value() for x in GObject.data['collision_objects']]

    # update current and last positions if the required time passed

    if time_passed:
        GObject.data['previous_position'] = GObject.data['current_position'].copy()
        GObject.data['current_position'] = real_values.copy()

    # if the simulator is under user control write position to a file and check mode every iteration

    if GObject.data['mode'] == 'USER':
        t = time() - GObject.data['start_time']
        line = ','.join([str(t)] + [str(val) for val in real_values])
        with open(user_input_fname, 'a') as f:
            f.write(line + '\n')
        mode = get_mode(GObject.data['id'])
        if mode is not None:
            GObject.data['mode'] = mode

    # if the simulator is under AI control

    else: # if GObject.data['mode'].startswith('AI'):

        # if the target is not set, ask for it immediately

        if GObject.data['is_target_set'] == False:

            # calculate velocity and request target

            target, _ = get_target(GObject.data['id'], GObject.data['current_position'], GObject.data['previous_position'], GObject.data['time_passed'], score_values[0], np.sum(collision_values))

            # in case of success

            if target is not None:
                GObject.data['integ_prev'] = 0
                GObject.data['target_position'] = target
                GObject.data['is_target_set'] = True
                GObject.data['in_target'] = [0 for _ in range(action_dim)]
                GObject.data['target_set_time'] = time()

        # if target has been already set

        else:

            # check if in target

            for i in range(action_dim):
                if np.abs(real_values[i] - GObject.data['target_position'][i]) < target_thr:
                    GObject.data['in_target'][i] = 1

            # schedule control change

            change_control = True

            # furthermore, if the time passed

            if time_passed:

                # check whether target or mode have changed

                target, mode = get_target(GObject.data['id'], GObject.data['current_position'], GObject.data['previous_position'], GObject.data['time_passed'], score_values[0], np.sum(collision_values))

                # process mode

                if mode is not None:
                    GObject.data['mode'] = mode

                    # if mode is USER, clean previous user input and nulify the state

                    if mode == 'USER':
                        open(user_input_fname, 'w').close()
                        print('\nUSER INPUT\n')
                        GObject.data['is_target_set'] = False
                        GObject.data['ground_volume'] = 0
                        GObject.data['start_time'] = time()
                        GObject.data['target_set_time'] = time()
                        GObject.data['last_time'] = time()
                        GObject.data['previous_position'] = [x.value() for x in GObject.data['component_objects']]
                        GObject.data['current_position'] = [x.value() for x in GObject.data['component_objects']]
                        GObject.data['time_passed'] = 0
                        GObject.data['target_position'] = None

                        # disable control change

                        change_control = False

                    elif mode == 'RESTART':
                        sys.exit(0)

                # processs target

                if target is not None:
                    GObject.data['integ_prev'] = 0
                    GObject.data['target_position'] = target
                    GObject.data['is_target_set'] = True
                    GObject.data['target_set_time'] = time()
                    GObject.data['in_target'] = [0 for _ in range(action_dim)]

                # apply PID control if it is enabled

                if change_control:
                    component_values, GObject.data['integ_prev'] = pid_controls(
                        GObject.data['current_position'],
                        GObject.data['previous_position'],
                        GObject.data['target_position'],
                        t_delta,
                        GObject.data['integ_prev']
                    )
                    for i in range(action_dim):
                        if GObject.data['in_target'][i] == 0:
                            GDict[component_controls[i]].setInputValue(component_values[i])
                        else:
                            GDict[component_controls[i]].setInputValue(0)