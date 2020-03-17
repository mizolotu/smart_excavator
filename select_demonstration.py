import subprocess, logging, winreg, requests, json, pandas, os, sys
import numpy as np
import os.path as osp

from threading import Thread
from flask import Flask, jsonify, request
from time import sleep, time

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

def split_data(points, times, cycle_min_duration=10):
    cycles = []
    cycle_times = []
    slew_twists = np.hstack([0, np.where((points[1:, 0] > 0) & (points[:-1, 0] < 0))[0]])
    for i in range(len(slew_twists) - 1):
        if times[slew_twists[i + 1]] - times[slew_twists[i]] > cycle_min_duration:
            cycle_time = times[slew_twists[i] - 1: slew_twists[i + 1] + 1]
            cycle = points[slew_twists[i] - 1: slew_twists[i + 1] + 1, :]
            cycles.append(cycle)
            cycle_times.append(cycle_time)
    return cycles, cycle_times

def retrieve_original_dataset(data_dir='data/raw', tkey='Time', xkeys=['ForceR_Slew r', 'Cylinder_BoomLift_L x', 'Cylinder_DipperArm x', 'Cylinder_Bucket x'], nsteps=32):

    # find data files

    files = []
    for f in os.listdir(data_dir):
        fpath = osp.join(data_dir, f)
        if osp.isfile(fpath):
            files.append(fpath)

    # retrieve data

    data = []
    for file in files:
        data.append([])
        p = pandas.read_csv(file, delimiter='\t', header=1)
        n = p.shape[0]
        points = np.zeros((n, len(xkeys)))
        for i,key in enumerate(xkeys):
            points[:, i] = p[key].values
        times = p[tkey].values
        pieces, piece_times = split_data(points, times)
        for piece,piece_time in zip(pieces,piece_times):
            x = np.zeros((nsteps, len(xkeys)))
            tmin = piece_time[0]
            tmax = piece_time[-1]
            t = np.arange(nsteps) * (tmax - tmin) / nsteps + tmin
            for j in range(piece.shape[1]):
                x[:, j] = np.interp(t, piece_time, piece[:, j])
    return data

def augment_data(sample, d_min=80, d_max=110):
    sample_aug = []
    dig_angle_orig = np.max([np.max(d[:, 0]) for d in sample])
    dig_angle_new = d_min + np.random.rand() * (d_max - d_min)
    dig_alpha = dig_angle_new / dig_angle_orig
    for j in range(len(sample)):
        a = sample[j][:, 0:1]
        x = sample[j][:, 1:]
        a_new = a
        a_new[a_new[:, 0] > 0] *= dig_alpha
        sample_aug.append(np.hstack([a_new, x]))
    return sample_aug

def resample(x, m):
    m_old = x.shape[0]
    n = x.shape[1]
    x_resampled = np.zeros((m, n))
    for i in range(n):
        x_resampled[:, i] = np.interp((np.arange(m) + 1) / m, (np.arange(m_old) + 1) / m_old, x[:, i])
    return x_resampled

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
    data_keys = ['x', 'l', 't', 'm', 'd', 'c']
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

def generate_demonstration_dataset(fname, mvs,
    n_series=10,
    http_url='http://127.0.0.1:5000',
    mode_uri='mode',
    dig_file = 'data/dig.txt',
    emp_file='data/emp.txt',
    n_steps = 8,
    delay=1.0, x_thr=[3.0, 5.0, 5.0, 5.0], t_thr=3.0, m_thr=10.0, m_max=1000.0, t_max=60.0, a_thr=3.0
):
    regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
    (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
    solverpath += r'\Bin\MeveaSolver.exe'
    winreg.CloseKey(regkey)
    solver_args = [solverpath, r'/mvs', mvs]

    best_mass = -np.inf
    best_lost = np.inf

    # main loop

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
        print('Started:')
        print(backend)
        idx = np.arange(len(data_orig))
        sample_orig = data_orig[np.random.choice(idx)]
        dsa = augment_data(sample_orig)
        dumped_last = 0

        T = []
        Xd = []
        Xe = []
        D = []
        C = []
        Digs = []
        Emps = []
        M = []

        for ci,cycle in enumerate(dsa):
            cycle_time_start = time()
            mass = np.zeros(cycle.shape[0])
            dig_target = None
            emp_target = None
            for i in range(cycle.shape[0]):
                target = cycle[i, :]
                post_target(target)
                in_target = np.zeros(4)
                if ci > 0 and i == cycle.shape[0] // 2:
                    D.append((backend['d'] - dumped_last) / m_max)
                    dumped_last = backend['d']
                t_start = time()
                while not np.all(in_target):
                    current = backend['x']
                    dist_to_x = np.abs(np.array(current) - target)
                    for j in range(4):
                        if dist_to_x[j] < x_thr[j]:
                            in_target[j] = 1
                    if (time() - t_start) > t_thr:
                        break
                cycle[i, :] = backend['x']
                mass[i] = backend['m']
                if mass[i] > m_thr and dig_target is None:
                    dig_target = backend['x']
                elif mass[i] < m_thr and dig_target is not None and emp_target is None and np.abs(dig_target[0] - backend['x'][0]) < a_thr:
                    emp_target = backend['x']

            # check the targets

            if dig_target is not None and emp_target is not None:
                dig_target_angle = dig_target[0]
                emp_target_angle = emp_target[0]
                didx = np.where((cycle[:, 0] > dig_target_angle - a_thr) & (cycle[:, 0] < dig_target_angle + a_thr))[0]
                eidx = np.where((cycle[:, 0] > emp_target_angle - a_thr) & (cycle[:, 0] < emp_target_angle + a_thr))[0]
                print(i, dig_target_angle, emp_target_angle, len(didx), len(eidx))

            # save the stats

            c = (cycle - np.ones((cycle.shape[0], 1)) * x_min) / (np.ones((cycle.shape[0], 1)) * (x_max - x_min + 1e-10))
            T.append((time() - cycle_time_start) / t_max)
            if dig_target is not None:
                Digs.append(c[didx, :])
                Xd.append((dig_target - x_min) / (x_max - x_min + 1e-10))
            if emp_target is not None:
                Emps.append(c[eidx, :])
                Xe.append((emp_target - x_min) / (x_max - x_min + 1e-10))

            C.append(c.reshape(4 * cycle.shape[0]))
            M.append(np.max(mass) / m_max)

        # for the last cycle we wait for few seconds to let the simulator to calculate the soil mass in the dumper

        sleep(3.0)
        D.append((backend['d'] - dumped_last) / m_max)

        # save data to the files

        if len(Xd) == n_cycles and len(Xe) == n_cycles:

            for ci in range(n_cycles):
                t = T[ci]
                xd = Xd[ci]
                xe = Xe[ci]
                d = D[ci]
                c = C[ci]
                m = M[ci]
                v = np.hstack([ci, xd, xe, t, m, d, c])
                line = ','.join([str(item) for item in v])
                with open(fname, 'a') as f:
                    f.write(line + '\n')
                print(ci, t, xd, xe, m, d)

            mass_array = np.hstack(M)
            idx = np.argmax(mass_array)
            if mass_array[idx] > best_mass:
                best_mass = mass_array[idx]
                dig = resample(Digs[idx], n_steps)
                with open(dig_file, 'w') as f:
                    for x in dig:
                        line = ','.join([str(item) for item in x]) + '\n'
                        f.write(line)

            lost_array = np.hstack([x - y for x,y in zip(M, D)])
            idx = np.argmin(lost_array)
            if lost_array[idx] < best_lost:
                emp = resample(Emps[idx], n_steps)
                with open(emp_file, 'w') as f:
                    for x in emp:
                        line = ','.join([str(item) for item in x]) + '\n'
                        f.write(line)

        else:
            print(Xd, Xe)

        # stop the software

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

    # process args

    if len(sys.argv) == 2:
        mvs = sys.argv[1]
    else:
        print('Please specify path to the excavator model!')
        sys.exit(1)

    # file name to save dataset

    fname = 'data/policy_data.txt'
    if not osp.exists(fname):
        open(fname, 'a').close()

    # original data

    n_cycles = 4
    data_orig_all = retrieve_original_dataset()
    data_orig = [series for series in data_orig_all if len(series) == n_cycles]
    x_min = np.array([-180.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])
    x_max = np.array([180.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])
    d_min = np.array([-0.05, -0.3, -0.5, -0.8])
    d_max = np.array([0.05, 0.3, 0.5, 0.8])
    m_max = 1000

    # start solver

    backend = {'ready': False, 'running': False, 'mode': 'AI_TRAIN', 'x': None, 'l': None, 't': None, 'y': None, 'm': None, 'd': None, 'c': None}
    th = Thread(target=generate_demonstration_dataset, args=(fname, mvs))
    th.setDaemon(True)
    th.start()

    # start http server

    print('Server starts')
    app.run(host='0.0.0.0')