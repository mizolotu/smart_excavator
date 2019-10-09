import pickle
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sequence_generator import TimePredictor, SequenceGenerator

# switch off CUDA
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Auxiliary functions

def get_recorded_data(level_file='data/level_points.pkl'):
    with open(level_file, 'rb') as f:
        level_points = pickle.load(f)
        key_points = pickle.load(f)
        key_stats = pickle.load(f)
    RX_raw = []
    deltas = []
    Deltas = []
    Times = []
    for level in level_points:
        RX_raw.append(level[0])
        deltas.append(level[0][1:, :] - level[0][:-1, :])
        for i in range(10):
            Deltas.append(level[0][i + 1:, :] - level[0][:-i - 1, :])
            Times.append(level[1][i + 1:] - level[1][:-i - 1])
    RX_raw = np.vstack(RX_raw)
    x_min = np.array([-360.0, 3.9024162648733514, 13.252630737652677, 16.775050853637147])  # np.min(RX_raw, axis=0)
    x_max = np.array([360.0, 812.0058600513476, 1011.7128949856826, 787.6024456729566])  # np.max(RX_raw, axis=0)
    print(x_min, x_max)
    ss = StandardScaler()
    mm = MinMaxScaler()
    X_ss = ss.fit_transform(RX_raw[:, 1:])
    mm.fit(X_ss)

    # find x_min and x_max for X deltas and distances

    deltas = np.abs(np.vstack(deltas))
    mm_d = MinMaxScaler()
    mm_d.fit(deltas)

    # generate train datasets for sequence generation and time prediction

    Deltas = np.abs(np.vstack(Deltas))
    ss_D = StandardScaler()
    TX = ss_D.fit_transform(Deltas)
    TY = np.vstack(Times)

    n_levels = len(level_points)
    n_points = len(level_points[0][2])
    n_features = len(level_points[0][2][0]) - 1

    DX = np.zeros((n_levels, n_features))
    DY = np.zeros((n_levels, n_points, n_features))

    Digs = []
    Emps = []

    for i in range(n_levels):
        dig = mm.transform(ss.transform(level_points[i][2][:, 1:]))
        DX[i, :] = dig[0, :]
        DY[i, :, :] = dig
        Digs.append(level_points[i][2])
        Emps.append(level_points[i][3])

    return TX, TY, mm_d, ss_D, DX, DY, ss, mm, Digs, Emps, key_points, key_stats, x_min, x_max

if __name__ == '__main__':

    # tf graphs and sessions

    sequence_generation_graph = tf.Graph()
    sequence_generation_session = tf.compat.v1.Session(graph=sequence_generation_graph)

    # sequence generator model file

    ed_model_file = 'models/trajectory_generator/dense'

    # train data

    TX, TY, mm_d, ss_D, DX, DY, ss, mm, Digs, Emps, key_points, key_stats, rx_min, rx_max = get_recorded_data()
    cycle_start_point, cycle_end_point, dig_start_point, dig_end_point, emp_start_point, emp_end_point = key_points
    dig_bucket_diff, emp_bucket_diff, dig_mean_angle, emp_mean_angle = key_stats

    n_levels = DY.shape[0]
    n_steps = DY.shape[1]
    n_features = DY.shape[2]

    # timer and digger models

    timer = TimePredictor(
        sequence_generation_graph,
        sequence_generation_session,
        n_features + 1
    )

    digger = SequenceGenerator(
        sequence_generation_graph,
        sequence_generation_session,
        n_features,
        n_steps
    )

    with sequence_generation_graph.as_default():
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sequence_generation_session, ed_model_file)
        except Exception as e:
            print(e)
            sequence_generation_session.run(tf.compat.v1.global_variables_initializer())
            timer.train(TX, TY)
            digger.train(DX, DY)
            saver.save(sequence_generation_session, ed_model_file, write_meta_graph=False)