import os
# Configuration file for parameters

# Data directories
data_df = '../data/metadata/neurips_2021_v4.csv'
data_dir = '../data/audio/'
plot_dir = '../outputs/plots/'

# Librosa settings
# Feature output directory
dir_out = '../outputs/features/'
rate = 8000
win_size = 30
step_size = 5
n_feat = 128
NFFT = 2048
n_hop = NFFT/4
frame_duration = n_hop/rate # Frame duration in ms

# Calculating window size based on desired min duration (sample chunks)
# default at 8000Hz: 2048 NFFT -> NFFT/4 for window size = hop length in librosa.
# Recommend lowering NFFT to 1024 so that the default hop length is 256 (or 32 ms).
# Then a win size of 60 produces 60x32 = 1.92 (s) chunks for training

min_duration = win_size * frame_duration # Change to match 1.92 (later)

# Create directories if they do not exist:
for directory in [plot_dir, dir_out]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)
