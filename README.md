# HumBugDB
Acoustic mosquito detection with Bayesian Neural Networks.

* Extract audio or features from our large-scale dataset on [Zenodo](https://zenodo.org/record/4904800).
* Jupyter notebooks to train, evaluate, and visualise BNNs and their associated uncertainty metrics for:
  * [Task 1: Mosquito Event Detection (MED)](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/mosquito_event_detection.md)
  * [Task 2: Mosquito Species Classification (MSC)](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/mosquito_species_classification.md)

By Ivan Kiskin. Contact `ivankiskin1@gmail.com` for enquiries or suggestions.

## General use instructions
This code is complementary to the paper: [*"HumBugDB: a large-scale acoustic mosquito dataset"*](https://openreview.net/forum?id=vhjsBtq9OxO) and the dataset on [Zenodo](https://zenodo.org/record/4904800).

See documentation in [the paper supplement](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/NeurIPS_2021_HumBugDB_Supplement.pdf) for:
* Section A: Licensing
* Section B: Code use, feature and model engineering
* Section C: Description and visualisations of metadata in `data/metadata/*.csv` 

Additional documentation for:
* [Code, feature, hyperparameter configuration](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/code_configuration.md)

## Installation instructions
Code compatible with both PyTorch and Keras (GPU). Installation instructions for PyTorch are given in `InstallationLogPyTorch.txt` which include the requirements to run all the code. Installation in PyTorch is simpler due to fewer dependency clashes. 

Keras requirements are given in `condarequirementsKeras.txt` and `piprequirementsKeras.txt`. 

After installation of requirements, the code can be run by cloning the repository:
`git clone https://github.com/HumBug-Mosquito/HumBugDB.git`.

Audio from the four-part-archive should be extracted from [Zenodo](https://zenodo.org/record/4904800) to `/data/audio/`.

## `v1.0` Reproducible Research release
If you wish to use the trained models, download the binaries from release `v1.0`, and place the models in `/outputs/models/keras/` or `/outputs/models/pytorch/` for the respective libraries. Note that for compatibility, features are required to be extracted with the default settings of the repository. We include the four models trained in the paper (and an additional improvement for Keras) for reproducibility in release `v1.0`. The plots of the paper are present in `outputs/plots/neurips_2021_reproducibility/` for the respective models and test sets. 
