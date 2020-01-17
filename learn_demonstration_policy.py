import tensorflow as tf
import numpy as np
import sys, os

def create_model(n_features, n_labels, n_layers=3, n_nodes=256):
    model = tf.keras.models.Sequential([tf.keras.Input(shape=(n_features,))])
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape((n_labels // n_features, n_features)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
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

    fname = 'data/policy_data.txt'
    data = np.loadtxt(fname, dtype=float, delimiter=',')
    action_dim = 4
    series_len = 16
    n_samples = data.shape[0]
    sample_len = data.shape[1]
    assert (sample_len - action_dim) % series_len == 0
    n_features = action_dim
    n_labels = data.shape[1] - action_dim
    x_train = data[:, :n_features]
    y_train = np.zeros((n_samples, series_len, action_dim))
    for i in range(n_samples):
        y_train[i, :, :] = data[i, n_features:].reshape(series_len, action_dim)
    print(x_train.shape, y_train.shape)

    # train model

    model = create_model(n_features, n_labels, series_len)
    epochs = 10000
    batch_size = 64
    checkpoint_path = "policies/demonstration/last.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=epochs)
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])