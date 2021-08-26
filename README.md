# HumBugDB
Acoustic mosquito detection with Bayesian Neural Networks.

* Extract audio or features from our large-scale dataset on [Zenodo](https://zenodo.org/record/4904800).
* Train BNNs for mosquito audio event detection
* Evaluate BNNs and their associated uncertainty metrics

By Ivan Kiskin. Contact `ivankiskin1@gmail.com` for enquiries or suggestions.

## General use instructions
This code is complementary to the paper: *"HumBugDB: a large-scale acoustic mosquito dataset"* [currently under review](https://neurips.cc/Conferences/2021/CallForDatasetsBenchmarks#:~:text=NeurIPS%202021%20Datasets%20and%20Benchmarks,how%20to%20improve%20dataset%20development.), and the dataset on [Zenodo](https://zenodo.org/record/4904800).

See documentation in `docs/NeurIPS_2021_HumBugDB_Supplement.pdf` for:
* Section A: Licensing
* Section B: Code use, feature and model engineering
* Section C: Description and visualisations of metadata in `data/metadata/*.csv` 

## Installation instructions
The code is written with compatibility for PyTorch and Keras (GPU), depending on the user's preference. Installation instructions for PyTorch are given in `InstallationLogPyTorch.txt` which include the requirements to run all the code. Installation in PyTorch is simpler due to fewer dependency clashes. 

Keras requirements are given in `condarequirementsKeras.txt` and `piprequirementsKeras.txt`. 

After installation of requirements, the code can be run by cloning the repository:
`git clone https://github.com/HumBug-Mosquito/HumBugDB.git`.

Audio from the four-part-archive should be extracted from [Zenodo](https://zenodo.org/record/4904800) to `/data/audio/`.

If you wish to use the trained models, download the binaries from release `v1.0`, and place the models in `/outputs/models/keras/` or `/outputs/models/pytorch/` for the respective libraries. Note that for compatibility, features are required to be extracted with the default settings of the repository.

## `v1.0` reproducible research release
We include the four models trained in the paper (and an additional improvement for Keras) for reproducibility in release `v1.0`. The plots of the paper are present in `outputs/plots/neurips_2021_reproducibility/` for the respective models and test sets. 

## Additional documentation
[Task 1: Mosquito Event Detection (MED)](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/mosquito_event_detection.md)

[Code, feature, hyperparameter configuration](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/code_configuration.md)
