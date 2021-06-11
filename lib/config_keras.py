import config
import os

epochs = 4
tau = 1.0
dropout = 0.2
batch_size = 32
lengthscale = 0.01

# Make output sub-directory for saving model
directory = os.path.join(config.model_dir, 'keras')
if not os.path.isdir(directory):
	os.mkdir(directory)
	print('Created directory:', directory)



