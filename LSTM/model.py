from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adagrad, Adam, Adadelta, RMSprop
from math import sqrt
import numpy as np

def r(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    N = predictions.shape[0]
    return (N*np.sum(np.multiply(predictions,targets)) - np.sum(predictions) * np.sum(targets))/np.sqrt((N*np.sum(predictions**2) - np.sum(predictions)**2)*(N*np.sum(targets**2) - np.sum(targets)**2))
    
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
def fit_lstm(train, batch_size, nb_epoch, neurons, optimizer, dropout=0.0):
    X, y = train[:, 0:-1], train[:, -1] 
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, dropout=dropout, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    if(optimizer == 'rmsprop'):
        opt = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    if(optimizer == 'adam'):
        opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    if(optimizer == 'adadelta'):
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    if(optimizer == 'adagrad'):
        opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    history = []
    for i in range(nb_epoch):
        hist = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        print("Epoch: {}, ".format(i+1),"loss: {}".format(hist.history['loss'][0]))
        history.append(hist.history['loss'][0])
        model.reset_states()
    return model, history
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]