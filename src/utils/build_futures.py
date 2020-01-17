import random
import numpy as np

def build_future_from_past_at_a_time(seq_model, ae_model, X, past_horizon=50, future_horizon=50, i=None):
    # how far into the past to look
    past = past_horizon
    # how far into the future to predict
    future = future_horizon
    # cannot look further into the future than what the model was trained for
    assert (past+future <= X.shape[1])

    if i == None:
        i = random.randint(0, X.shape[0]-1)
    # initialize hidden and cell states of LSTM
    h, c = np.array([np.zeros(1024)]), np.array([np.zeros(1024)])
    
    # initialize input to model
    Y_hat, Y = [], []
    x = X[i, :past, :]
    y = X[i, past:past+future, :]
    Y.append(y)

    # roll out future frames a single timestep at a time
    x = np.expand_dims(x, axis=0)
    for j in range(future):
        x_new, h_new, c_new = seq_model.forward.predict([x, h, c])
        if j == 0:
            x_new = x_new[:, past-1:, :]
        Y_hat.append(x_new)
        x, h, c = x_new, h_new, c_new
    Y_hat = np.array(Y_hat)
    Y = np.array(Y)
    
    # decoded future frames into interpretable images
    Y_hat = np.reshape(Y_hat, (1, future, 64))
    X_decoded = ae_model.decoder.predict(X[i, :past, :])
    Y_hat_decoded = ae_model.decode_series(Y_hat)
    Y_decoded = ae_model.decode_series(Y)
    
    return X_decoded, Y_decoded[0], Y_hat_decoded[0], i

def build_future_from_past(seq_model, ae_model, X, past_horizon=50, future_horizon=50, i=None):
    # how far into the past to look
    past = past_horizon
    # how far into the future to predict
    future = future_horizon
    # cannot look further into the future than what the model was trained for
    assert (past+future <= X.shape[1])

    if i == None:
        i = random.randint(0, X.shape[0]-1)
    
    # initialize input to model
    _X, Y = [], []
    x = X[i, :past, :]
    y = None
    if future == 0:
        y = x
    else:
        y = X[i, past:past+future, :]
    _X.append(x)
    Y.append(y)
    _X = np.array(_X)
    Y = np.array(Y)

    # build output
    embedding = seq_model.encoder.predict(_X)
    Y_hat = seq_model.decoder.predict(embedding)
    X_decoded = ae_model.decode_series(_X)
    # result = np.concatenate((X, Y_hat), axis=1)
    Y_hat_decoded = ae_model.decode_series(Y_hat)
    # result = np.concatenate((X, Y), axis=1)
    Y_decoded = ae_model.decode_series(Y)
    return X_decoded[0], Y_decoded[0], Y_hat_decoded[0], embedding, i
