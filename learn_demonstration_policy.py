import tensorflow as tf
import numpy as np
import sys, os

from matplotlib import pyplot as pp

def create_model(n_steps, n_features, n_nodes=256):

    # create train model

    model_tr = tf.keras.models.Sequential([tf.keras.Input(shape=(1, n_features), batch_size=n_steps)])
    model_tr.add(tf.keras.layers.LSTM(n_nodes, activation='relu', return_sequences=False, stateful=True))
    model_tr.add(tf.keras.layers.Dropout(0.5))
    model_tr.add(tf.keras.layers.Dense(n_features, activation='sigmoid'))
    model_tr.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model_tr.summary())

    # create inference model

    model_inf = tf.keras.models.Sequential([tf.keras.Input(shape=(1, n_features), batch_size=1)])
    model_inf.add(tf.keras.layers.LSTM(n_nodes, activation='relu', return_sequences=False, stateful=True))
    model_inf.add(tf.keras.layers.Dense(n_features, activation='sigmoid'))
    model_inf.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model_inf.summary())

    return model_tr, model_inf

def validate_and_plot(model, x0, y, w=4, h=4):
    p = np.zeros((x0.shape[0], y.shape[1], y.shape[2]))
    p[:, 0, :]  = x0
    for i in range(x0.shape[0]):
        for j in range(y.shape[1] - 1):
            pij = model.predict(p[i:i+1, j:j+1, :])
            print(i,j, pij)
            p[i, j + 1, :] = pij
    fig, axs = pp.subplots(w, h)
    fig.set_size_inches(18.5, 10.5)
    for i in range(w):
        for j in range(h):
            idx = i * w + j
            axs[i, j].plot(p[idx, :, :])
            axs[i, j].plot(y[idx, :, :], '--')
    fig.savefig(validation_fig, dpi=fig.dpi)

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

    x = np.zeros((n_samples, n_steps, n_features))
    y = np.zeros((n_samples, n_steps, n_features))
    for i in range(n_samples):
        traj = np.ones((n_steps + 1, 1)).dot(data[i:i+1, :n_features]) - data[i, n_features:].reshape(series_len, action_dim)
        x[i, :, :] = traj[:-1, :]
        y[i, :, :] = traj[1:, :]
    print(x.shape, y.shape)

    # train model

    n_validation = 16
    epochs = 10000
    batch_size = 32
    checkpoint_prefix = "policies/demonstration/last.ckpt"
    validation_fig = "policies/demonstration/validation.png"

    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    x_train = x[n_validation:, :, :]
    y_train = y[n_validation:, :, :]
    x_val = x[:n_validation, :, :]
    y_val = y[:n_validation, :, :]
    model_tr, model_inf = create_model(n_steps, n_features)
    try:
        model_inf.load_weights(checkpoint_prefix)
    except Exception as e:
        print(e)
        for epoch in range(epochs):
            for i in range(n_samples - n_validation):
                h = model_tr.fit(x_train[i:i+1, :, :].reshape(n_steps, 1, n_features), y_train[i:i+1, :, :].reshape(n_steps, n_features), verbose=False)
                model_tr.reset_states()
            if epoch % (epochs // 100) == 0:
                print(epoch, h.history)
                model_inf.set_weights(model_tr.get_weights())
                validate_and_plot(model_inf, x_val[:, 0, :], y_val)
                model_inf.save_weights(checkpoint_prefix)

    validate_and_plot(model_inf, x_val[:, 0, :], y_val)