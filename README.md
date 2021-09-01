# HumBugDB
Acoustic mosquito detection with Bayesian Neural Networks.

* Extract audio or features from our large-scale dataset on [Zenodo](https://zenodo.org/record/4904800).
* This repository outlines two key example use cases for the data:
  * [Task 1: Mosquito Event Detection (MED)](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/mosquito_event_detection.md)
  * [Task 2: Mosquito Species Classification (MSC)](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/mosquito_species_classification.md)

## General use instructions
This code is complementary to the paper: [*"HumBugDB: a large-scale acoustic mosquito dataset"*](https://openreview.net/forum?id=vhjsBtq9OxO) and the dataset on [Zenodo](https://zenodo.org/record/4904800).

See documentation in [the paper supplement](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/NeurIPS_2021_HumBugDB_Supplement.pdf) for:
* Section A: Licensing
* Section B: Code use, feature and model engineering
* Section C: Description and visualisations of metadata in `data/metadata/*.csv` 

Additional documentation for:
* [Code, feature, hyperparameter configuration](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/code_configuration.md)

## Installation instructions
Code compatible with both PyTorch and Keras (GPU).
* Installation instructions for PyTorch: `InstallationLogPyTorch.txt` which include the requirements to run all the code. 
* Keras requirements are given in `condarequirementsKeras.txt` and `piprequirementsKeras.txt`. 

After installation of requirements:
* ```     
  git clone https://github.com/HumBug-Mosquito/HumBugDB.git
* Extract audio from four-part-archive to [Zenodo](https://zenodo.org/record/4904800) to `/data/audio/`.

## Contact
Developed by Ivan Kiskin of [MLRG](https://www.robots.ox.ac.uk/~parg/) University of Oxford. Contact `ivankiskin1@gmail.com` for enquiries or suggestions.
Follow our Twitter on [@OxHumBug](https://twitter.com/oxhumbug) and visit our [HumBug website](https://humbug.ox.ac.uk/) for updates on the overall HumBug project.

