import json, logging, os, sys, argparse
import numpy as np

from excavator_env import ExcavatorEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn, demonstrate
from baselines import logger
from threading import Thread
from flask import Flask, jsonify, request

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

def create_env(id, policy):
    return lambda : ExcavatorEnv(id, policy)

@app.route('/register')
def register():
    global envs
    eid = None
    for key in range(len(envs)):
        if not envs[key]['backend_assigned']:
            envs[key]['backend_assigned'] = True
            eid = key
            break
    return jsonify({'id': eid})

@app.route('/id', methods=['GET', 'POST'])
def assign_reset():
    global envs
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    key = jdata['id']
    if request.method == 'POST':
        envs[key]['backend_assigned'] = False
    assigned = envs[key]['backend_assigned']
    return jsonify({'assigned': assigned})

@app.route('/targets', methods=['GET', 'POST'])
def get_targets():
    global envs
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    key = jdata['id']
    if request.method == 'POST':
        if None not in jdata['targets']:
            envs[key]['target_list'] = jdata['targets']
            envs[key]['mode'] = 'AI'
            print_banner()
    return jsonify({'targets': envs[key]['target_list']})

@app.route('/mode', methods=['GET', 'POST'])
def mode():
    global envs
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    id = jdata['id']
    if request.method == 'GET':
        envs[id]['backend_running'] = True
    elif request.method == 'POST':
        mode = jdata['mode']
        envs[id]['mode'] = mode
        if mode == 'RESTART':
            envs[id]['backend_running'] = False
    return jsonify({'mode': envs[id]['mode']})

@app.route('/p_target', methods=['GET', 'POST'])
def target():
    global envs
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    id = jdata['id']
    data_keys = ['x', 'l', 't', 'm', 'd', 'c']
    if request.method == 'GET':
        for key in data_keys:
            envs[id][key] = jdata[key]
        if envs[id]['y'] is not None:
            y = envs[id]['y'].copy()
        else:
            y = None
        mode = envs[id]['mode']
        return jsonify({'y': y, 'mode': mode})
    elif request.method == 'POST':
        if jdata['y'] is not None:
            envs[id]['y'] = jdata['y']
        data = {}
        for key in data_keys:
            data[key] = envs[id][key]
        data['backend_running'] = envs[id]['backend_running']
        return jsonify(data)

def print_banner(fname='banner.txt'):
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        print(line, end = '')
    print('\n')

if __name__ == '__main__':

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the excavator model.', required=True)
    parser.add_argument('-t', '--task', default='test', help='Task: train or test.', choices=['train', 'test'])
    parser.add_argument('-p', '--policy', default='residual', help='Policy: ppo or residual.', choices=['ppo', 'residual'])
    parser.add_argument('-n', '--nenvs', type=int, default=2, help='Number of environments for training.')
    args = parser.parse_args()

    # disable cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # configure logger

    log_dir = 'policies/{0}'.format(args.policy)
    logger.configure(log_dir)

    # min and max values

    x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
    x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
    d_min = np.array([-0.05, -0.5, -0.5, -0.8])
    d_max = np.array([0.05, 0.5, 0.5, 0.8])

    # specify random target for training

    dig_target = np.array([0.72, 0.56, 0.82, 0.27]) * (x_max - x_min) + x_min
    emp_target = np.array([0.29, 0.71, 0.39, 0.22]) * (x_max - x_min) + x_min
    train_targets = [dig_target.tolist(), emp_target.tolist()]

    # environment raw data

    if args.task == 'train':
        envs = [
            {'backend_assigned': False, 'backend_running': False, 'mode': 'AI', 'target_list': train_targets, 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'd': None, 'c': None}
            for _ in range(args.nenvs)
        ]
    elif args.task == 'test':
        envs = [
            {'backend_assigned': False, 'backend_running': False, 'mode': 'USER', 'target_list': [None, None], 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'd': None, 'c': None}
        ]
    else:
        print('What?')
        sys.exit(1)

    # create environments

    nsteps = 8
    nupdates = 1000000
    env_fns = [create_env(key, args.policy) for key in range(len(envs))]

    if args.task == 'train':
        env = SubprocVecEnv(env_fns)
    elif args.task == 'test':
        env = SubprocVecEnv(env_fns)

    if args.task == 'train':
        learn_th = Thread(target=learn, args=('mlp', env, nsteps, nsteps * nupdates // len(envs), args.model))
        learn_th.start()
    elif args.task == 'test':
        test_th = Thread(target=demonstrate, args=('mlp', env, nsteps, args.model))
        test_th.start()
    else:
        print('What?')
        sys.exit(1)

    # start http server

    print('Server starts')
    app.run(host='0.0.0.0')