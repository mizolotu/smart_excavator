import tensorflow as tf
import numpy as np
import sys, os

from matplotlib import pyplot as pp

def create_model(n_steps, n_features, n_nodes=256):

    # create train model

    model_tr = tf.keras.models.Sequential([tf.keras.Input(shape=(n_steps, n_features))])
    model_tr.add(tf.keras.layers.Masking(mask_value=0., input_shape=(n_steps, n_features)))
    model_tr.add(tf.keras.layers.LSTM(n_nodes, activation='relu'))
    model_tr.add(tf.keras.layers.Dropout(0.5))
    model_tr.add(tf.keras.layers.Dense(n_features, activation='sigmoid'))
    model_tr.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model_tr.summary())

    return model_tr

def validate_and_plot(model, x0, y, w=4, h=4):
    p = np.zeros((x0.shape[0] // n_steps, n_steps, y.shape[1]))
    r = np.zeros((x0.shape[0] // n_steps, n_steps, y.shape[1]))
    for i in range(x0.shape[0] // n_steps):
        r[i, :, :] = y[i * n_steps : (i+1) * n_steps, :]
        x_ = x0[i * n_steps, :, :]
        for j in range(n_steps):
            pij = model.predict(x_.reshape(1, n_steps, n_features))
            p[i, j, :] = pij
            x_ = np.vstack([x_[1:, :], pij])
    fig, axs = pp.subplots(w, h)
    fig.set_size_inches(18.5, 10.5)
    for i in range(w):
        for j in range(h):
            idx = i * w + j
            axs[i, j].plot(p[idx, :, :])
            axs[i, j].plot(r[idx, :, :], '--')
    fig.savefig(validation_fig, dpi=fig.dpi)
    pp.close(fig)

def create_traj_model(n_features, n_labels, n_layers=2, n_nodes=2048):
    model = tf.keras.models.Sequential([tf.keras.Input(shape=(n_features,))])
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape((n_labels // n_features, n_features)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

def create_lstm_model(n_features, n_labels, series_len, n_layers = 2, n_nodes = 256):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(series_len, n_features)))
    for i in range(n_layers - 1):
        model.add(tf.keras.layers.LSTM(n_nodes, activation='relu', return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LSTM(n_nodes, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    # disable cuda if needed

    if'cpu' in sys.argv[1:]:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #  get data

    action_dim = 4
    fname = 'data/policy_data.txt'
    data = np.loadtxt(fname, dtype=float, delimiter=',')
    data = np.hstack([data[:, :action_dim], data[:, action_dim + 1:]])  # remove mass
    n_samples = data.shape[0]
    sample_len = data.shape[1]
    assert (sample_len - action_dim) % action_dim == 0
    series_len = (sample_len - action_dim) // action_dim
    n_features = action_dim
    n_steps = (data.shape[1] - action_dim) // n_features - 1

    x = np.zeros((n_samples * n_steps, n_steps, n_features))
    y = np.zeros((n_samples * n_steps, n_features))
    for i in range(n_samples):
        target_minus_traj = np.ones((n_steps + 1, 1)).dot(data[i:i+1, :n_features]) - data[i, n_features:].reshape(series_len, action_dim)
        traj = data[i, n_features:].reshape(series_len, action_dim)
        for j in range(n_steps):
            x[i * n_steps + j, n_steps-j-1:n_steps, :] = target_minus_traj[:j+1, :]
            y[i * n_steps + j, :] = traj[j+1, :]
    print(x.shape, y.shape)

    # train model

    n_validation = 16
    epochs = 10000
    batch_size = 31
    checkpoint_prefix = "policies/demonstration/last.ckpt"
    validation_fig = "policies/demonstration/validation.png"

    x_train = x[n_validation * n_steps:, :, :]
    y_train = y[n_validation * n_steps:, :]
    x_val = x[:n_validation * n_steps, :, :]
    y_val = y[:n_validation* n_steps, :]
    model = create_model(n_steps, n_features)
    try:
        model.load_weights(checkpoint_prefix)
    except Exception as e:
        print(e)
        for epoch in range(epochs):
            h = model.fit(x_train, y_train, verbose=False, batch_size=batch_size)
            if epoch % (epochs // 100) == 0:
                print(epoch, h.history)
                validate_and_plot(model, x_val, y_val)
        model.save_weights(checkpoint_prefix)

    validate_and_plot(model, x_val[:, 0, :], y_val)