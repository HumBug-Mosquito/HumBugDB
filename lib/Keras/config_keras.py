import config
import os

# General:
# Adadelta optimizer configuration
learning_rate=1.0
rho=0.95
epsilon=1e-07

# BNN weight regularisation:
tau = 1.0
dropout = 0.2

# Settings for mosquito event detection:
epochs = 4
batch_size = 32
lengthscale = 0.01

# Settings for multi-species as in paper:
batch_size = 128 
epochs = 80




# Make output sub-directory for saving model
directory = os.path.join(config.model_dir, 'keras')
if not os.path.isdir(directory):
	os.mkdir(directory)
	print('Created directory:', directory)



