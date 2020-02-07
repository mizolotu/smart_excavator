import json, logging, os

from excavator_env import ExcavatorEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn as learn_ppo
from baselines.ddpg.ddpg import learn as learn_ddpg
from baselines import logger
from threading import Thread
from flask import Flask, jsonify, request

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

def create_env(id):
    return lambda : ExcavatorEnv(id)

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
    return jsonify({'assigned': envs[key]['backend_assigned']})

@app.route('/targets')
def get_targets():
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    key = jdata['id']
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
    data_keys = ['x', 'l', 't', 'm', 'c']
    if request.method == 'GET':
        for key in data_keys:
            envs[id][key] = jdata[key]
        if envs[id]['y'] is not None:
            y = envs[id]['y'].copy()
            envs[id]['y'] = None # every time an excavator requests the target, we nulify it, this guarantees that the excavator operates with the fresh target
        else:
            y = None
        mode = envs[id]['mode']
        return jsonify({'y': y, 'mode': mode})
    elif request.method == 'POST':
        envs[id]['y'] = jdata['y']
        data = {}
        for key in data_keys:
            data[key] = envs[id][key]
        data['backend_running'] = envs[id]['backend_running']
        return jsonify(data)


if __name__ == '__main__':

    # disable cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # configure logger

    log_dir = 'policies/human+ppo'
    logger.configure(log_dir)

    # target lists

    targets = [[72.77621012856298, 448.1429081568389, 706.6441233973571, 257.47653181221153], [-76, 608, 413, 232]]

    # environment raw data

    envs = [
        {'backend_assigned': False, 'backend_running': False, 'mode': 'AI_TRAIN', 'target_list': targets, 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'c': None},
        {'backend_assigned': False, 'backend_running': False, 'mode': 'AI_TRAIN', 'target_list': targets, 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'c': None}
    ]

    # create environments

    load_path = 'policies/human/checkpoints/last'
    nsteps = 8
    nupdates = 1000
    env_fns = [create_env(key) for key in range(len(envs))]
    env = SubprocVecEnv(env_fns)
    reset_th = Thread(target=learn_ppo, args=('mlp', env, len(envs) * nsteps * nupdates, load_path))
    reset_th.start()

    # start http server

    print('Server starts')
    app.run(host='0.0.0.0')