import tensorflow as tf
import numpy as np

def create_mlp_model(n_features, n_labels, n_layers = 3, n_nodes = 256):
    model = tf.keras.models.Sequential([tf.keras.Input(shape=(n_features,))])
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_nodes, activation='relu'))
    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def create_model(n_features, n_labels, n_layers = 2, n_nodes = 256):
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

    #  get data

    fname = 'data/policy_data.txt'
    data = np.loadtxt(fname, dtype=float, delimiter=',')
    action_dim = 4
    series_len = 14
    n_samples = data.shape[0]
    assert n_samples % series_len == 0
    n_features = data.shape[1] - action_dim
    #x_train = data[:, :n_features]
    x_train = np.zeros((n_samples, series_len, n_features))
    m_train = np.zeros((n_samples, series_len), dtype=bool)
    for i in range(0, n_samples, series_len):
        series = data[i : i + series_len, :]
        for j in range(series_len):
            x_train[i+j, :, :] = series[:, :n_features]
            x_train[i+j, j+1:, :] = 0.
            m_train[i+j, :j+1] = True
    print(x_train[0:series_len])
    y_train = data[:, n_features:]

    # train model

    model = create_model(n_features, action_dim)
    epochs = 10000
    batch_size = 64
    checkpoint_path = "policies/demonstration/last.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=10)
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])