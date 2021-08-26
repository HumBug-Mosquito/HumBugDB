## Data configuration `config.py`
Specify the metadata (csv) location in `data_df`, with the location of the raw wave files in `data_dir`. The desired output for the features is set by `dir_out`. Model objects will be saved to `outputs/models/pytorch/` or `outputs/models/keras/`. By default, metadata is stored in `/data/metadata/`, audio files are found in `/data/audio/`, and features in `/outputs/features`.

## Features
A more detailed discussion on feature transformations is given in Appendix B of the supplement in `/docs/`.

### Feat. A
Features with default configuration from the VGGish [GitHub](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_input.py) intended for use with VGGish: 64 log-mel spectrogram coefficients using 96 feature frames of 10 ms duration. To compare feature sets fairly, predictions are aggregated over neighbouring windows to create outputs over 1.92 second windows as used in Feat. B. The behaviour of these features can be tweaked to user preference within `lib/vggish/vggish_input.py`, `lib/feat_vggish.py` and their dependencies.

### Feat. B
The feature extraction uses log-mel features with `librosa`, configurable in `config.py` with the sample rate `rate`, to which data is re-sampled on loading, a window size `win_size` which determines the size of a training window (in number of _feature_ windows), `step_size`, which determines the step size taken by the window, `NFFT`, and `n_hop`, which are parameters for the core STFT transform upon which log-mel feature extraction is based. Finally, `n_feat` determines the number of mel-frequency bands.

In `librosa`, we can calculate the value of `win_size` to achieve a user's desired `duration` for a label as follows:

`win_size` = `duration` / `frame_duration`, where `frame_duration` = `n_hop`/`rate`. Librosa uses a default `hop_length` of `NFFT/4`.
The default values in `config.py` are optimised for `rate` = 8000 with  `win_size` = 30, `NFFT` = 2048, `n_hop` = `default`,  to achieve a label duration of 30 * 2048/4 * (1/8000) = 1.92 (s). 

## Model hyperparameters
### Keras `config_keras.py`
`tau`,`lengthscale`, are parameters used for L2 weight regularization supplied in `runKeras.py`. `dropout` controls the dropout rate, `batch_size` controls the batch size, and `epochs`, set the number of epochs to train. 
### PyTorch `config_pytorch.py`
Settings to change the learning rate, `lr`, the number of maximum overrun steps for a particular training criteria `max_overrun`, the number of `epochs`, and the `batch_size`. Whether to use pre-trained weights for the models is configured by the boolean `pretrained`. `n_classes` sets the number of classes to use with the `CrossEntropyLoss()` function in PyTorch.
