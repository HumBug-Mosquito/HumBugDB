import os
# Configuration file for parameters

# Data directories
data_df = os.path.join(os.path.pardir, 'data', 'metadata', 'neurips_2021_zenodo_0_0_1.csv')
# data_df = os.path.join(os.path.pardir, 'data', 'metadata', 'db_10_06_21_inc_false_positives.csv')
data_dir = os.path.join(os.path.pardir, 'data', 'audio')
plot_dir = os.path.join(os.path.pardir, 'outputs',  'plots')
model_dir = os.path.join(os.path.pardir, 'outputs', 'models') # Model sub-directory created in config_keras or config_pytorch

# Librosa settings
# Feature output directory
# sub-directory for mosquito event_detection
dir_out_MED = os.path.join(os.path.pardir, 'outputs', 'features', 'MED')
# sub-directory for mosquito species classification
dir_out_MSC = os.path.join(os.path.pardir, 'outputs', 'features', 'MSC')  
rate = 8000
win_size = 30
step_size = 5
n_feat = 128
NFFT = 2048
n_hop = NFFT/4
frame_duration = n_hop/rate # Frame duration in ms
# Normalisation
norm_per_sample = True


# Calculating window size based on desired min duration (sample chunks)
# default at 8000Hz: 2048 NFFT -> NFFT/4 for window size = hop length in librosa.
# Recommend lowering NFFT to 1024 so that the default hop length is 256 (or 32 ms).
# Then a win size of 60 produces 60x32 = 1.92 (s) chunks for training

min_duration = win_size * frame_duration # Change to match 1.92 (later)

# Create directories if they do not exist:
for directory in [plot_dir, dir_out_MED, dir_out_MSC, model_dir]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)
