# Task 1: Mosquito Event Detection
This task is the detection of acoustic mosquito events of any species from their background surroundings, such as other insects, speech, urban, and rural noise. A decsription of the data collection process, the data splits, and baseline performance is given in the paper [*HumBugDB: A Large-scale Acoustic Mosquito
Dataset*
](https://openreview.net/pdf?id=vhjsBtq9OxO). 


## Notebook: `main.ipynb`

* Select the library as `PyTorch` or `Keras` from the notebook. This will in turn import the necessary functions for use with any model.
* Select the features to use: `FeatA` will use VGGish log-mel features, `FeatB` will use baseline 128 log-mel features as described [code_configuration.md](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/code_configuration.md)
* Partition the data as laid out in the notebook.
* Train model with `train_model()`. `train_model()` shares a common interface for Keras and PyTorch. Supply `X_train, y_train` to train and checkpoint according to the training accuracy. Supply `X_train, y_train, X_val, y_val` to use validation data for checkpointing. Models will be saved with their epoch name, and training and/or validation losses. If using Keras, the function supports the model defined in `runKeras.py`. For PyTorch, supply the model object as an input argument. Models are defined in `runTorch.py`.
  * Keras models are configured in `config_keras.py`. Leave as default to reproduce the training strategy used in the paper.
  * PyTorch models are configured in `config_pytorch.py`. Leave as default to reproduce the training strategy used in the paper.
* Produce results for the two test sets from `get_results()`. By default, windows are handled appropriately to produce outputs over the same temporal window of 1.92 seconds for fair comparison. The different evaluation pipelines are handled in `runKeras.py` and `runPyTorch.py`.
* (Optional) configure data input and output directories in `config.py`. 
* (Optional) configure or modify the feature creation in `feat_vggish.py`, `vggish_input.py` for Feat. A, and in `config.py` for Feat. B.

## Model configuration
The hyperparameter settings used for MED are:
* `config_keras`: `epochs=4`, `batch_size=32`, `lengthscale=0.01`, `dropout=0.2`
* `config_pytorch`: `epochs=50`, `lr=0.0015`, `dropout=0.2`, `pretrained=True`

## Trained models
The results of the model training are included in this repository for the purpose of reproducibility, or if you would like to use standalone models for your own applications. Consider that the input pipeline of audio and feature processing has to be matched identically for correct function. 
* If you wish to use a trained model in a detection pipeline which is streamlined for *deployment*, please visit our other repo [MozzBNN](https://github.com/HumBug-Mosquito/MozzBNN). 
* If you wish to use any of the trained models from the experiments reported in Section 5.1 of the paper, you may download any of the following models which are linked to [GitHub release v2.0](https://github.com/HumBug-Mosquito/HumBugDB/releases/tag/2.0).

| model                                                          | feat | model_class |
|----------------------------------------------------------------|------|-------------|
| neurips_2021_humbugdb_vgg_A.pth                              | A    | vgg         |
| neurips_2021_humbugdb_mozzbnnv2_A.hdf5 | A    | mozzbnnv2   |
| neurips_2021_humbugdb_resnet18_A.pth                               | A    | resnet18    |
| neurips_2021_humbugdb_resnet50_A.pth                               | A    | resnet50    |
| neurips_2021_humbugdb_vgg_B.pth                               | B    | vgg         |
| neurips_2021_humbugdb_mozzbnnv2_B.hdf5                     | B    | mozzbnnv2   |
| neurips_2021_humbugdb_resnet18_B.pth                        | B    | resnet18    |
| neurips_2021_humbugdb_resnet50_B.pth                               | B    | resnet50    |
