import pickle, json, os, requests, logging
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sequence_generator import AngleDetector

from generate_levels import resample_cycle_points

if __name__ == '__main__':

    # switch off CUDA

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # tf graphs and sessions

    angle_detection_graph = tf.Graph()
    angle_detection_session = tf.compat.v1.Session(graph=angle_detection_graph)

    # angle detection model file

    a_model_file = 'models/angle_detection/dense'

    # train data

    level_file = 'data/level_points.pkl'

    with open(level_file, 'rb') as f:
        level_points = pickle.load(f)
        key_points = pickle.load(f)
        key_stats = pickle.load(f)
        stats = pickle.load(f)

    X = []
    A = []
    X1 = []
    A1 = []
    A2 = []

    for i in range(len(level_points)):
        print(i)
        px = level_points[i][0]
        pt = level_points[i][1]
        s = stats[i]
        dig_idx = s[2:4]
        emp_idx = s[4:6]
        dig_a = np.mean(px[dig_idx[0] : dig_idx[1], 0])
        emp_a = np.mean(px[emp_idx[0]: emp_idx[1], 0])
        l = emp_idx[1] - dig_idx[0] + 1
        for j in range(px.shape[0] - l):
            r = np.random.rand() * 100 - 50
            x = px[j:, :]
            t = pt[j:, :]
            points_resampled, time_step = resample_cycle_points({'x': x, 't': t}, time_step=0.25)
            points_resampled[:, 0] += r
            dig_a_r = dig_a + r
            emp_a_r = emp_a + r
            a_min = np.min(points_resampled[:, 0])
            a_max = np.max(points_resampled[:, 0])
            a_step = 0.5
            a_window = 1
            X1.append([])
            A1.append(dig_a_r)
            A2.append(emp_a_r)
            for a in np.arange(a_min, a_max, a_step):
                idx = np.where((points_resampled[:, 0] > a - a_window) & (points_resampled[:, 0] < a + a_window))[0]
                if len(idx) > 0:
                    feature_vector = np.vstack([
                        np.mean(points_resampled[idx, :], axis=0),
                        np.std(points_resampled[idx, :], axis=0),
                        np.min(points_resampled[idx, :], axis=0),
                        np.max(points_resampled[idx, :], axis=0),
                    ])
                    X.append(feature_vector[:, 1:])
                    X1[-1].append(feature_vector)
                    if a > dig_a_r - a_window and a < dig_a_r + a_window:
                        A.append([0, 1, 0])
                    elif a > emp_a_r - a_window and a < emp_a_r + a_window:
                        A.append([0, 0, 1])
                    else:
                        A.append([1, 0, 0])

    # standardize features

    x_min = np.array([3.9024162648733514, 13.252630737652677, 16.775050853637147])
    x_max = np.array([812.0058600513476, 1011.7128949856826, 787.6024456729566])
    X_to_std = np.vstack([np.vstack(X), x_min, x_max])
    mm = MinMaxScaler().fit(X_to_std)

    n_steps = np.max([x.shape[0] for x in X])
    n_features = 3
    n_classes = 3

    n_train = len(X)
    X_train = np.zeros((n_train, n_steps, n_features))
    Y_train = np.zeros((n_train, 3))

    n_test0 = len(X1)
    n_test1 = np.max([len(x) for x in X1])
    X_test = np.zeros((n_test0, n_test1, n_steps, n_features))

    print(X_train.shape, Y_train.shape)
    print(X_test.shape)

    for i in range(n_train):
        n = X[i].shape[0]
        X_train[i, :n, :] = mm.transform(X[i])
        Y_train[i, :] = A[i]

    for i in range(n_test0):
        n = len(X1[i])
        for j in range(n):
            m = X1[i][j].shape[0]
            X_test[i, j, :m, :] = mm.transform(X1[i][j][:, 1:])

    angler = AngleDetector(
        angle_detection_graph,
        angle_detection_session,
        n_steps,
        n_features,
        n_classes,
        lr=0.0001,
    )

    with angle_detection_graph.as_default():
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(angle_detection_session, a_model_file)
        except Exception as e:
            print(e)
            angle_detection_session.run(tf.compat.v1.global_variables_initializer())
            angler.train(X_train, Y_train, epochs=100000, batch=10000)
            saver.save(angle_detection_session, a_model_file, write_meta_graph=False)

    da = []
    df = 0
    dp = []
    ea = []
    ep = []
    ef = 0
    for i in range(n_test0):
        n = len(X1[i])
        p = angler.predict(X_test[i, :n, :, :])
        l = np.argmax(p, axis=1)
        d_idx = np.array(np.where(l == 1)[0], dtype=int)
        e_idx = np.array(np.where(l == 2)[0], dtype=int)
        if len(d_idx) > 0:
            d = []
            for idx in d_idx:
                d.append(X1[i][idx][0, 0])
            dp.append(np.mean(d))
            da.append(A1[idx])
            print(d, A1[idx])
        else:
            df += 1
        if len(e_idx) > 0:
            e = []
            for idx in e_idx:
                e.append(X1[i][idx][0, 0])
            ep.append(np.mean(e))
            ea.append(A2[i])
        else:
            ef += 1
    d_error = np.abs(np.array(da) - np.array(dp))
    e_error = np.abs(np.array(ea) - np.array(ep))
    print(np.min(d_error), np.max(d_error), np.mean(d_error), float(df) / n_test0)
    print(np.min(e_error), np.max(e_error), np.mean(e_error), float(ef) / n_test0)