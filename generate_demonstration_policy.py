import pickle, subprocess, logging, winreg, requests, json
import numpy as np
from threading import Thread

from flask import Flask, jsonify, request
from time import sleep, time

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

def retrieve_original_dataset(data_file='data/level_points.pkl', nsteps=32):
    with open(data_file, 'rb') as f:
        levels = pickle.load(f)
    data = []
    t_last = np.inf
    for li,level in enumerate(levels):
        t = level[1][:, 0]
        x = level[0]
        if t[1] < t_last:
            t_last = t[-1]
            data.append([])
        step = len(t) // nsteps
        t_line = np.array([t[i * step] for i in np.arange(nsteps)])
        m = len(t_line)
        n = x.shape[1]
        p = np.zeros((m, n))
        for j in range(x.shape[1]):
            p[:, j] = np.interp(t_line, t, x[:, j])
        data[-1].append(p)
    return data

def augment_data(sample, a_min=75, a_max=115):
    sample_aug = []
    dig_angle_orig = np.max([np.max(d[:, 0]) for d in sample])
    dig_angle_new = a_min + np.random.rand() * (a_max - a_min)
    alpha = dig_angle_new / dig_angle_orig
    for j in range(len(sample)):
        a = sample[j][:, 0:1]
        x = sample[j][:, 1:]
        a_new = a
        a_new[a_new[:, 0] > 0] *= alpha
        sample_aug.append(np.hstack([a_new, x]))
    return sample_aug

def start_simulator(solver_args, http_url='http://127.0.0.1:5000', n_attempts=30, uri='ready'):
    url = '{0}/{1}'.format(http_url, uri)
    print('Trying to start solver...')
    ready = False
    while not ready:
        attempt = 0
        registered = False
        subprocess.Popen(solver_args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        while not registered:
            try:
                j = requests.get(url).json()
                registered = j['ready']
            except Exception as e:
                print(e)
            attempt += 1
            if attempt >= n_attempts:
                break
            sleep(1.0)
        if registered:
            ready = True
            print('Solver has successfully started!')
        else:
            print('Could not start solver :( Trying again...')

@app.route('/register')
def register(eid=0):
    global backend
    if not backend['ready']:
        backend['ready'] = True
    return jsonify({'id': eid})

@app.route('/ready', methods=['GET', 'POST'])
def assign_reset():
    global backend
    if request.method == 'POST':
        backend['ready'] = False
    return jsonify({'ready': backend['ready']})

@app.route('/mode', methods=['GET', 'POST'])
def mode():
    global backend
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    if request.method == 'GET':
        backend['running'] = True
    elif request.method == 'POST':
        mode = jdata['mode']
        backend['mode'] = mode
        if mode == 'RESTART':
            backend['running'] = False
    return jsonify({'mode': backend['mode']})

@app.route('/p_target', methods=['GET', 'POST'])
def target():
    global backend
    data = request.data.decode('utf-8')
    jdata = json.loads(data)
    data_keys = ['x', 'l', 't', 'm', 'c']
    if request.method == 'GET':
        for key in data_keys:
            backend[key] = jdata[key]
        if backend['y'] is not None:
            y = backend['y'].copy()
            backend['y'] = None # every time an excavator requests the target, we nulify it, this guarantees that the excavator operates with the fresh target
        else:
            y = None
        mode = backend['mode']
        return jsonify({'y': y, 'mode': mode})
    elif request.method == 'POST':
        backend['y'] = jdata['y']
        data = {}
        for key in data_keys:
            data[key] = backend[key]
        data['running'] = backend['running']
        return jsonify(data)

def generate_demonstration_dataset(fname, n_series=1000, mws = 'C:\\Users\\iotli\\PycharmProjects\\SmartExcavator\\mws\\env.mws', http_url='http://127.0.0.1:5000', mode_uri='mode', delay=1.0, a_thr=3.0, x_thr=5.0, t_thr=3.0, m_thr=50.0, m_max=1000):
    regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
    (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
    solverpath += r'\Bin\MeveaSolver.exe'
    winreg.CloseKey(regkey)
    solver_args = [solverpath, r'/loadmws', mws, r'/saveplots', r'/silent']
    for si in range(n_series):
        start_simulator(solver_args)
        ready = False
        while not ready:
            jdata = post_target()
            if jdata is not None:
                if jdata['running'] and jdata['x'] is not None and jdata['l'] is not None and jdata['m'] is not None:
                    ready = True
                else:
                    sleep(delay)
        requests.post('{0}/{1}'.format(http_url, mode_uri), json={'mode': 'AI_TRAIN'}).json()
        print('started')
        print(backend)
        sample_orig = np.random.choice(data_orig)
        dsa = augment_data(sample_orig)
        for ci,cycle in enumerate(dsa):
            print(ci)
            dig_angle = None
            bucket_close_target_idx = None
            dig_target = None
            bucket_max = 0
            mass = np.zeros(cycle.shape[0])
            for i in range(cycle.shape[0]):
                target = cycle[i, :]
                post_target(target)
                in_target = np.zeros(4)
                mass[i] = backend['m']
                if mass[i] > m_thr and dig_angle is None:
                    dig_target = backend['x']
                    dig_angle = backend['x'][0]
                if dig_angle is not None and np.abs(backend['x'][0] - dig_angle) <= a_thr and backend['x'][3] > bucket_max:
                    bucket_max = backend['x'][3]
                    bucket_close_target_idx = i
                t_start = time()
                while not np.all(in_target):
                    current = backend['x']
                    dist_to_x = np.abs(np.array(current) - target)
                    for i in range(4):
                        if dist_to_x[i] < x_thr:
                            in_target[i] = 1
                    if (time() - t_start) > t_thr:
                        break
            if dig_target is not None:
                print('Dig here: {0}'.format(dig_target))
                t = (dig_target - x_min) / (x_max - x_min + 1e-10)
                c = (cycle - np.ones((cycle.shape[0], 1)) * x_min) / (np.ones((cycle.shape[0], 1)) * (x_max - x_min + 1e-10))
                v = c.reshape(4 * cycle.shape[0])
                m = mass[bucket_close_target_idx] / m_max
                x = np.hstack([t, m, v])
                line = ','.join([str(item) for item in x])
                with open(fname, 'a') as f:
                    f.write(line + '\n')

                #for i in range(cycle.shape[0] - 2):
                #    a = (cycle[i + 2,:] - x_min) / (x_max - x_min + 1e-10)
                #    c = (cycle[i + 1,:] - x_min) / (x_max - x_min + 1e-10)
                #    l = (cycle[i,:] - x_min) / (x_max - x_min + 1e-10)
                #    m = mass[i] / m_max
                #    print(dig_angle, bucket_close_target_idx, t, a, m)
                #    x = np.hstack([t - l, t - c, m, a]).tolist()
                #    line =','.join([str(item) for item in x])
                #    with open(fname, 'a') as f:
                #        f.write(line + '\n')

        requests.post('{0}/{1}'.format(http_url, mode_uri), json={'mode': 'RESTART'}).json()
        print('stopped')
        print(backend)

def post_target(target=None, http_url='http://127.0.0.1:5000', uri='p_target'):
    url = '{0}/{1}'.format(http_url, uri)
    if target is not None:
        target = target.tolist()
    try:
        jdata = requests.post(url, json={'y': target}).json()
    except Exception as e:
        print(e)
        jdata = None
    return jdata

if __name__ == '__main__':

    # number of time series (number of solver restarts)

    n_series = 100

    # file name to save dataset

    fname = 'data/policy_data.txt'
    #open(fname, 'w').close()

    # original data

    data_orig = retrieve_original_dataset()
    x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
    x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
    m_max = 1000

    # start solver

    backend = {'ready': False, 'running': False, 'mode': 'AI_TRAIN', 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'c': None}
    th = Thread(target=generate_demonstration_dataset, args=(fname, n_series))
    th.setDaemon(True)
    th.start()

    # start http server

    print('Server starts')
    app.run(host='0.0.0.0')