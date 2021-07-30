import tensorflow as tf
from Keras import config_keras # local module
import config
# Deep learning
# Keras-related imports
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution1D, MaxPooling2D, Convolution2D
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.callbacks import ModelCheckpoint, RemoteMonitor, EarlyStopping
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
import os
from datetime import datetime

def train_model(X_train, y_train, X_val=None, y_val=None, class_weight=None, start_from=None):


	y_train = tf.keras.utils.to_categorical(y_train)
	if y_val is not None:
		y_val = tf.keras.utils.to_categorical(y_val)


	################################ CONVOLUTIONAL NEURAL NETWORK ################################
	## NN parameters

	input_shape = (1, X_train.shape[2], X_train.shape[-1]) # Input shape for channels_first

	# BNN parameters
	dropout=config_keras.dropout  
	# Regularise
	tau = config_keras.tau
	lengthscale = config_keras.lengthscale
	reg = lengthscale**2 * (1 - dropout) / (2. * len(X_train) * tau)

	W_regularizer=l2(reg)  # regularisation used in layers

	# Initialise optimiser for consistent results across Keras/TF versions
	opt = Adadelta(learning_rate=config_keras.learning_rate, rho=config_keras.rho, epsilon=config_keras.epsilon) 

	model = Sequential()
	n_dense = 128
	nb_classes = y_train.shape[1]
	# number of convolutional filters
	nb_conv_filters = 32
	# num_hidden = 236
	nb_conv_filters_2 = 64


	model.add(Conv2D(nb_conv_filters, kernel_size = (3,3),
	     activation = 'relu', padding = 'valid', strides = 1,
	     input_shape = input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Lambda(lambda x: K.dropout(x,level=dropout)))


	model.add(Conv2D(nb_conv_filters_2, kernel_size = (3,3),
	     activation = 'relu', padding = 'valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Lambda(lambda x: K.dropout(x,level=dropout)))

	model.add(Conv2D(nb_conv_filters_2, kernel_size = (3,3),
	     activation = 'relu', padding = 'valid'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Lambda(lambda x: K.dropout(x,level=dropout)))

	

	# # model.add(Dropout(0.2))
	model.add(Conv2D(nb_conv_filters_2, kernel_size = (3,3),
	     activation = 'relu', padding = 'valid'))
	model.add(Lambda(lambda x: K.dropout(x,level=dropout)))

	# model.add(Conv2D(nb_conv_filters_2, kernel_size = (3,3),
	#      activation = 'relu', padding = 'valid'))
	# model.add(Lambda(lambda x: K.dropout(x,level=dropout)))

	model.add(Flatten())
	# # Shared between MLP and CNN:
	model.add(Dense(n_dense, activation='relu'))
	model.add(Lambda(lambda x: K.dropout(x,level=dropout)))


	model.add(Dense(nb_classes, activation='softmax', kernel_regularizer=l2(reg)))
	model.compile(loss='categorical_crossentropy',
	                optimizer=opt,
	                metrics=['accuracy'])


	if start_from is not None:
		model = load_model(start_from)
		print('Starting from model', start_from)

    # if checkpoint_name is not None:
    # 	os.path.join(os.path.pardir, 'models', 'keras', checkpoint_name)

	if X_val is None:
		metric = 'accuracy'
		val_data = None
	else:
		metric = 'val_accuracy'
		val_data = (X_val, y_val)

	model_name = 'Win_' + str(config.win_size) + '_Stride_' + str(config.step_size) + '_'
	model_name = model_name + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-e{epoch:02d}' + metric + '{' + metric + ':.4f}.hdf5'
	# model_name = model_name + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-e{epoch:02d}.hdf5'
	checkpoint_filepath = os.path.join(config.model_dir, 'keras',  model_name)
	model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=False,
		monitor=metric,
		mode='max',
		save_best_only=False)

	model.fit(x=X_train, y=y_train, batch_size=config_keras.batch_size, epochs=config_keras.epochs, verbose=1,
		validation_data=val_data,
		shuffle=True, class_weight=class_weight, sample_weight=None, initial_epoch=0,
		steps_per_epoch=None, validation_steps=None, callbacks=[model_checkpoint_callback])


	
	return model

def evaluate_model(model, X_test, y_test, n_samples):
	all_y_pred = []
	for n in range(n_samples):
		all_y_pred.append(model.predict(X_test))
	return all_y_pred

def load_model(filepath):
	model = tf.keras.models.load_model(filepath, custom_objects={"dropout": config_keras.dropout})
	return model