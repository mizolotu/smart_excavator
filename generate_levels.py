import pickle, csv
import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_data_from_file(file_name, data_names):
    data = {}
    column_ids = {}
    for key in data_names:
        column_ids[key] = []
        data[key] = []
    if file_name in data_files:
        with open(file_name, 'rt', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader)
            column_names = next(reader)
            column_names = [name.strip() for name in column_names]
            for key in data_names:
                for j in range(len(data_names[key])):
                    if data_names[key][j] in column_names:
                        column_ids[key].append(column_names.index(data_names[key][j]))
            for row in reader:
                np_row = np.array(row)
                np_row = np_row.astype(np.float)
                for key in data_names:
                    data[key].append(np_row[column_ids[key]])
    return data

def extract_points(data):
    timestamps = np.vstack(data['timestamp'])
    measurements = np.vstack(data['measurements'])
    points = {'x': measurements, 't': timestamps}
    return points

def extract_excavator_cycles(points, cycle_min_duration=10):
    cycles = []
    slew_twists = np.hstack([0, np.where((points['x'][1:, 0] > 0) & (points['x'][:-1, 0] < 0))[0]])
    for i in range(len(slew_twists) - 1):
        if points['t'][slew_twists[i + 1]] - points['t'][slew_twists[i]] > cycle_min_duration:
            cycle = {
                'x': points['x'][slew_twists[i] - 1: slew_twists[i + 1] + 1, :],
                't': points['t'][slew_twists[i] - 1: slew_twists[i + 1] + 1],  # - points['t'][slew_twists[i]],
            }
            cycles.append(cycle)
    return cycles

def partition_cycle(cycle, t_eps=10, n_bins=50):
    ss = StandardScaler()
    mm = MinMaxScaler()
    x_ss = ss.fit_transform(cycle['x'])
    x_mm = mm.fit_transform(x_ss)
    cycle_s = {'t': cycle['t'], 'x': x_mm}
    slew_sign = np.sign(cycle_s['x'][t_eps, 0] - cycle['x'][0, 0])
    slew_hist = np.histogram(cycle_s['x'][:, 0], bins=n_bins)
    bin_size = slew_hist[1][1] - slew_hist[1][0]
    half = int(n_bins / 2)
    if slew_sign == 1:
        digging_slew = slew_hist[1][half + np.argmax(slew_hist[0][half:])]
        emptying_slew = slew_hist[1][np.argmax(slew_hist[0][:half])]
    else:
        # TO DO
        pass

    bin_alpha = 1
    digging_idx = np.where(
        (
            cycle_s['x'][:, 0] >= digging_slew - bin_alpha * bin_size
        ) & (
            cycle_s['x'][:, 0] <= digging_slew + bin_size * bin_alpha
        )
    )[0]
    emptying_idx = np.where(
        (
            cycle_s['x'][:, 0] >= emptying_slew - bin_alpha * bin_size
        ) & (
            cycle_s['x'][:, 0] <= emptying_slew + bin_size * bin_alpha
        )
    )[0]
    return digging_idx, emptying_idx, np.array([0, digging_idx[0]]), np.array([digging_idx[-1], emptying_idx[0]])

def analyze_sequence(xd, xe, a_step=0.25, a_radius=0.5):
    bucket_angle_dif = []
    for x in np.arange(0, np.max(xd[:, 0]), a_step):
        idx = np.where((xd[:, 0] >= x - a_radius) & (xd[:, 0] < x + a_radius))[0]
        if len(idx) > 0:
            bucket_angle = xd[idx, 3]
            diff = np.max(bucket_angle) - np.min(bucket_angle)
            bucket_angle_dif.append([x, diff])
    if len(bucket_angle_dif) > 0:
        bucket_angle_dif = np.array(bucket_angle_dif)
        dig_ad = bucket_angle_dif[np.argmax(bucket_angle_dif[:, 1]), :]
    else:
        dig_ad = np.zeros(2)
    bucket_angle_dif = []
    for x in np.arange(0, np.min(xe[:, 0]), - a_step):
        idx = np.where((xe[:, 0] >= x - a_radius) & (xe[:, 0] < x + a_radius))[0]
        if len(idx) > 0:
            bucket_angle = xe[idx, 3]
            diff = np.max(bucket_angle) - np.min(bucket_angle)
            bucket_angle_dif.append([x, diff])
    if len(bucket_angle_dif) > 0:
        bucket_angle_dif = np.array(bucket_angle_dif)
        emp_ad = bucket_angle_dif[np.argmax(bucket_angle_dif[:, 1]), :]
    else:
        emp_ad = np.zeros(2)
    return dig_ad, emp_ad

def resample_cycle_points(points, n_steps=None, time_step=None):
    if time_step == None and n_steps != None:
        time_step = (points['t'][-1] - points['t'][0]) / n_steps
    t = np.arange(points['t'][0], points['t'][-1], time_step)
    n_features = np.shape(points['x'])[1]
    points_resampled = np.zeros((len(t), n_features))
    for i in range(n_features):
        points_resampled[:, i] = np.interp(t, points['t'].reshape(len(points['t'])),  points['x'][:, i].reshape(len(points['t'])))
    return points_resampled, time_step

def generate_user_input(fname, cycle):
    with open(fname, 'w') as f:
        for t, vector in zip(cycle['t'], cycle['x']):
            vector_str = [str(t[0])] + [str(item) for item in vector]
            line = ','.join(vector_str)
            f.write(line + '\n')

if __name__ == '__main__':
    data_dir = 'data'
    data_files = [
        join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.txt')
    ]
    data_names = {
        'timestamp': ['Time'],
        'control': ['Input_Slew RealValue', 'Input_BoomLift RealValue', 'Input_DipperArm RealValue',
                    'Input_Bucket RealValue'],
        'measurements': ['ForceR_Slew r', 'Cylinder_BoomLift_L x', 'Cylinder_DipperArm x', 'Cylinder_Bucket x']
    }
    n_files = len(data_files)

    user_input_file = 'user_input.txt'

    levels = []
    X = []
    dig_start_point = []
    emp_start_point = []
    dig_end_point = []
    emp_end_point = []
    cycle_start_point = []
    cycle_end_point = []
    dig_a = []
    emp_a = []
    for fi in range(n_files):
        data = get_data_from_file(data_files[fi], data_names)
        points = extract_points(data)
        cycles = extract_excavator_cycles(points)
        print(fi, len(cycles))
        for i, cycle in enumerate(cycles):
            if i == 0:
                generate_user_input(user_input_file, cycle)
            dig_i, emp_i, to_dig_i, to_emp_i = partition_cycle(cycle)
            task_i = [
                dig_i,
                emp_i,
                to_dig_i,
                to_emp_i
            ]
            n_steps = 32
            tasks = []
            for t_i in task_i:
                tasks_i = {'t': cycle['t'][t_i], 'x': cycle['x'][t_i]}
                tasks.append(resample_cycle_points(tasks_i, np.minimum(len(t_i), n_steps)))
                print(tasks[-1][0].shape, tasks[-1][1])
            dig = tasks[0][0]
            emp = tasks[1][0]
            d_ad, e_ad = analyze_sequence(dig, emp)
            dig_a.append(d_ad)
            emp_a.append(e_ad)
            dig_start_point.append(dig[0, :])
            dig_end_point.append(dig[-1, :])
            emp_start_point.append(emp[0, :])
            emp_end_point.append(emp[-1, :])
            cycle_start_point.append(cycle['x'][0, :])
            cycle_end_point.append(cycle['x'][-1, :])
            #points_resampled, time_step = resample_cycle_points(cycle, n_steps=128)
            #X.append(points_resampled)
            #levels.append((points_resampled, time_step, tasks[0], tasks[1]))
            levels.append((cycle['x'], cycle['t'], tasks))

    # dig and emp start points

    dig_start_point = np.mean(np.vstack(dig_start_point), axis=0)
    dig_end_point = np.mean(np.vstack(dig_end_point), axis=0)
    emp_start_point = np.mean(np.vstack(emp_start_point), axis=0)
    emp_end_point = np.mean(np.vstack(emp_end_point), axis=0)
    cycle_start_point = np.mean(np.vstack(cycle_start_point), axis=0)
    cycle_end_point = np.mean(np.vstack(cycle_end_point), axis=0)
    key_points = (cycle_start_point, cycle_end_point, dig_start_point, dig_end_point, emp_start_point, emp_end_point)

    dig_a = np.vstack(dig_a)
    emp_a = np.vstack(emp_a)
    key_stats = (np.min(dig_a[:,1]), np.min(emp_a[:,1]), np.mean(dig_a[:, 0]), np.mean(emp_a[:, 0]))

    level_file = 'level_points.pkl'
    with open(join(data_dir, level_file), 'wb') as f:
        pickle.dump(levels, f)
        pickle.dump(key_points, f)
        pickle.dump(key_stats, f)