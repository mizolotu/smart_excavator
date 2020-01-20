import tensorflow as tf
import numpy as np
import sys, os

def create_model(n_features, n_labels, n_layers=2, n_nodes=256):
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
    #optimizer = tf.keras.optimizers.Adam(3e-4)
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
    n_labels = data.shape[1] - action_dim
    x_train = data[:, :n_features]
    y_train = np.zeros((n_samples, series_len, action_dim))
    for i in range(n_samples):
        y_train[i, :, :] = data[i, n_features:].reshape(series_len, action_dim)
    print(x_train.shape, y_train.shape)

    # train model

    model = create_model(n_features, n_labels)
    epochs = 10000
    batch_size = 32
    checkpoint_prefix = "policies/demonstration/last.ckpt"
    try:
        model.load_weights(checkpoint_prefix)
    except Exception as e:
        print(e)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        model.save_weights(checkpoint_prefix)

    y_test = model.predict(x_train)
    d_train = np.zeros(n_samples)
    d_test = np.zeros(n_samples)
    for i in range(n_samples):
        d_train[i] = np.min(np.sqrt(np.sum((np.ones((series_len, 1)) * x_train[i, :] - y_train[i, :, :]) ** 2, axis=1)))
        d_test[i] = np.min(np.sqrt(np.sum((np.ones((series_len, 1)) * x_train[i, :] - y_test[i, :, :]) ** 2, axis=1)))
    print(n_samples, np.linalg.norm(d_train - d_test) / n_samples)

    #t = np.array([[0.70215614, 0.54973221, 0.69446078, 0.31226379]])
    #p = model.predict(t)[0]
    #for i in range(4):
    #    pp.plot(p[:,i])
    #    pp.plot()
    #    pp.show()