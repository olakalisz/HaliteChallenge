import os
import tensorflow as tf
import numpy as np

from tsmlstarterbot.common import PER_SHIP_FEATURES, PER_SHIP_ACTIONS, PER_MOVE_FEATURES

from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam

def make_model():
	# Build network
	model = Sequential()
	model.add(Dense(64,input_dim=PER_SHIP_FEATURES, kernel_initializer='random_uniform'))
	model.add(Activation('elu'))
	model.add(Dense(64, kernel_initializer='random_uniform'))
	model.add(Activation('elu'))
	model.add(Dense(PER_MOVE_FEATURES, kernel_initializer='random_uniform'))
	model.add(Activation('elu'))

	adam = Adam(lr=0.00001)
	model.compile(optimizer=adam, loss='mse')
	return model

def main():
	model = make_model()
	model.fit(X, y)
	model.predict()
	model.save('test.out')