import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from LSTM.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def build_model(self, input_dim, optimizer):
		timer = Timer()
		timer.start()
		self.model.add(LSTM(1, dropout=0.3, input_shape=(1, 30, input_dim), stateful=True))
		# self.model.add(LSTM((1), input_shape=(30,1), return_sequences=True))
		# self.model.add(LSTM((1), input_shape=(30,1), return_sequences=True))
		# self.model.add(LSTM((1), input_shape=(30,1), return_sequences=True))

		self.model.add(Dense(30))

		self.model.compile(loss='mse', optimizer=optimizer)

		print('[Model] Model Compiled')
		timer.stop()

	def train_generator(self, x, y, epochs, batch_size, steps_per_epoch, save_dir, data, optimizer):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		for i in range(epochs):
			history = self.model.fit(
				x, 
				y,
				epochs=1,
				batch_size=1
			)
			self.model.reset_states()
		yhat=self.model.predict(x)
		self.model.save(save_dir +  '%s-%s.h5' % (data, optimizer))
		timer.stop()
		return history,yhat