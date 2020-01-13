import sys, requests, json
import numpy as np
from time import time, sleep
from itertools import cycle

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
    ]
}

# Component names

components = [p.split(' ')[0] for p in data_names['measurements']]
parameters = [p.split(' ')[1] for p in data_names['measurements']]
score_components = [p.split(' ')[0] for p in data_names['scores']]
score_parameters = [p.split(' ')[1] for p in data_names['scores']]
component_controls = [p.split(' ')[0] for p in data_names['controls']]

# gains and thresholds

p_gain = 1 # P controller gain
p_controller_frequency = 1000 # Hz
eps = 1e-10 # to prevent dividing by zero

# dimensions

state_dim = 4
action_dim = 4

# targets

x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
idx = 0
target_min = [np.nan for i in range(action_dim)]
target_min[idx] = x_min[idx]
target_max = [np.nan for _ in range(action_dim)]
target_max[idx] = x_max[idx]
targets = [target_min, target_max]
targets_cycle = cycle(targets)
target_thr = 10

def p_controls(current, target):
    current = np.array(current)
    target = np.array(target)
    p = target - current
    controls = -p * p_gain
    return np.nan_to_num(controls)


##### I N I T  S C R I P T #####


def initScript():

    # try to register

    GObject.data['component_objects'] = [
        GSolver.getParameter(components[index], parameters[index]) for index in range(len(components))
    ]
    GObject.data['score_objects'] = [
        GSolver.getParameter(score_components[index], score_parameters[index]) for index in range(len(score_components))
    ]

    # initial state

    GObject.data['start_time'] = time()
    GObject.data['target_set_time'] = time()
    GObject.data['last_time'] = time()
    GObject.data['start_position'] = [x.value() for x in GObject.data['component_objects']]
    GObject.data['current_position'] = [x.value() for x in GObject.data['component_objects']]
    GObject.data['average_time'] = 0
    GObject.data['n_experiments'] = 0
    GObject.data['time_passed'] = 0
    GObject.data['target_position'] = next(targets_cycle)
    GObject.data['min_distance_to_target'] = np.inf


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

    GObject.data['current_position'] = [x.value() for x in GObject.data['component_objects']]
    target_dist = np.linalg.norm([(x - y) for x,y in zip(GObject.data['target_position'], GObject.data['current_position']) if np.isnan(x) == False])
    if target_dist < GObject.data['min_distance_to_target']:
        GObject.data['min_distance_to_target'] = target_dist
    if GObject.data['min_distance_to_target'] <= target_thr:
        dist_covered = np.linalg.norm([(x - y) for x, y in zip(GObject.data['start_position'], GObject.data['current_position'])])
        time_spent = time() - GObject.data['target_set_time']
        GObject.data['average_time'] += time_spent
        GObject.data['n_experiments'] += 1
        print('Speed: {0} after {1} experiments'.format(GObject.data['average_time'] / GObject.data['n_experiments'], GObject.data['n_experiments']))
        GObject.data['target_position'] = next(targets_cycle)
        GObject.data['target_set_time'] = time()
        GObject.data['min_distance_to_target'] = np.inf
    if time_passed:
        component_values = p_controls(GObject.data['current_position'], GObject.data['target_position'])
        for i in range(action_dim):
            GDict[component_controls[i]].setInputValue(component_values[i])