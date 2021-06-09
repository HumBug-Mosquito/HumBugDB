# HumBugDB
Acoustic mosquito detection code with Bayesian Neural Networks

# General use instructions
This code is complementary to the paper: HumBugDB: a large-scale acoustic mosquito dataset [currently under review](https://neurips.cc/Conferences/2021/CallForDatasetsBenchmarks#:~:text=NeurIPS%202021%20Datasets%20and%20Benchmarks,how%20to%20improve%20dataset%20development.), and the dataset on [Zenodo](https://zenodo.org/record/4904800).

We include the supplement in `docs/` which contains Licensing in Section A, Code use instructions in Section B, and explanations and visualisations of the metadata present in the `csv` file `config.data_df` found in `data/metadata/` in Section C.

# Installation instructions
The code is written with compatibility for PyTorch and Keras (GPU), depending on the user's preference. Installation instructions for PyTorch are given in `InstallationLogPyTorch.txt` which include the requirements to run all the code. Installation in PyTorch is simpler due to fewer dependency clashes. 

Keras requirements are given in `condarequirementsKeras.txt` and `piprequirementsKeras.txt`. 

After installation of requirements, the code can be run by cloning the repository:
`git clone https://github.com/HumBug-Mosquito/HumBugDB.git`

Audio from the four-part-archive should be extracted from [Zenodo](https://zenodo.org/record/4904800) to `/data/audio/`

# Reproducibility
We include the three models trained in the paper (@reviewers: WIP) for reproducibility in the first release of this code (due to the large sizes of the models). The plots of the paper are present in `outputs/plots/` for the respective models and test sets. 

# Code structure
`notebooks` contain `main.ipynb` and `supplement.ipynb` for the main code and supplementary code used to generate Figures in the Appendix.

Settings are specified in `config.py` and `config_pytorch.py` or `config_keras.py` which are located in `/lib`. Functions are imported from data and feature processing code in `/lib/feat_util.py`, model training in `/lib/runTorch.py` or `/lib/runKeras.py` and evaluation in `/lib/evaluate.py`.

`main.ipynb` provides the interface to partition data, extract features, train a BNN model in either PyTorch or Keras and evaluate its accuracy, precision-recall, confusion matrices and uncertainty metrics. Settings are specified in `config.py` and `config_pytorch.py` or `config_keras.py` which are located in `/lib/`. Functions are imported from data and feature processing code in `/lib/feat_util.py`, model training in `/lib/runTorch.py` or `/lib/runKeras.py` and evaluation in `/lib/evaluate.py`.

### Data configuration `config.py`
Specify the metadata (csv) location in `data_df`, with the location of the raw wave files in `data_dir`. The desired output for the features is set by `dir_out`. Model objects will be saved to `outputs/models/pytorch/` or `outputs/models/keras/`. By default, metadata is stored in `/data/metadata/`, audio files are found in `/data/audio/`, and features in `/outputs/features`.

The feature extraction uses log-mel features with `librosa`, configurable in `config.py` with the sample rate `rate`, to which data is re-sampled on loading, a window size `win_size` which determines the size of a training window (in number of _feature_ windows), `step_size`, which determines the step size taken by the window, `NFFT`, and `n_hop`, which are parameters for the core STFT transform upon which log-mel feature extraction is based. Finally, `n_feat` determines the number of mel-frequency bands.

In `librosa`, we can calculate the value of `win_size` to achieve a user's desired `duration` for a label as follows:

`win_size` = `duration` / `frame_duration`, where `frame_duration` = `n_hop`/`rate`. Librosa uses a default `hop_length` of `NFFT/4`.
The default values in `config.py` are optimised for `rate` = 8000 with  `win_size` = 30, `NFFT` = 2048, `n_hop` = `default`,  to achieve a label duration of 30 * 2048/4 * (1/8000) = 1.92 (s). A discussion on feature transformations is given in Appendix B of the supplement in `/docs/`.

### PyTorch `config_pytorch.py`
`config_pytorch.py` incldues settings to change the learning rate, `lr`, the number of maximum overrun steps for a particular training criteria `max_overrun`, the number of `epochs`, and the `batch_size`. The type of training method used can be written in `train_model.py`, which by default supports saving the best epoch models for either the training accuracy, `best_train_acc`, or validation accuracy, `best_val_acc`, if supplied to `train_model`.

### Keras `config_keras.py`
`tau = 1.0`,`lengthscale = 0.01`, are parameters used for $l2$ weight regularization supplied in lines 35-37 of `runKeras.py`. `dropout = 0.2` controls the dropout rate,`validation_split = 0.2`, is the fraction of data supplied as validation to the model callbacks in `model.fit`, line 105. `batch_size` controls the batch size, and `epochs`, set the number of epochs to train. Note the slight difference between the two packages in the way validation data is passed to the model training.
