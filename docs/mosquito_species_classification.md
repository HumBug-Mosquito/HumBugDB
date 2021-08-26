# Task 2: Mosquito Species Classification

## Notebook: `species_classification.ipynb`

* Select the library as `PyTorch` or `Keras` from the notebook. This will in turn import the necessary functions for use with any model.
* Select the features to use: `FeatA` will use VGGish log-mel features, `FeatB` will use baseline 128 log-mel features as described [code_configuration.md](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/code_configuration.md)
* Partition the data as laid out in the notebook. By default, the 8 most populated species of `country=Tanzania`, `location_type=cup` data is used. Outside of the task, you may load any other location and location type, and select the species you wish to train/test on by supplying `classes` to `get_train_test_multispecies()`.
* `get_train_test_multispecies()` will retrieve features from the dataframe as specified by `df_all`. The random seeds used to partition by recordings in this paper were 5, 10, 42, 100. Data by default is split by recordings, as each recording represents a unique mosquito. Note that this is only true for this dataset in particular. For other data, there may be repeated recordings of indivduals, so we advise to train and test on different experiment sources. The 8 recording sources are given in the main paper and [HumBugDB supplement](https://github.com/HumBug-Mosquito/HumBugDB/blob/devel/docs/NeurIPS_2021_HumBugDB_Supplement.pdf). _Future release may add a new field of metadata which will tag unique mosquito IDs to simplify this process. If you wish for early access please open an issue and we will do our best to accommodate._
* Train model with `train_model()`. `train_model()` shares a common interface for Keras and PyTorch. Supply `X_train, y_train` to train and checkpoint according to the training accuracy. Supply `X_train, y_train, X_val, y_val` to use validation data for checkpointing. Models will be saved with their epoch name, and training and/or validation losses. If using Keras, the function supports the model defined in `runKeras.py`. For PyTorch, supply the model object as an input argument. Models are defined in `runTorchMulticlass.py`.
  * Keras models are configured in `config_keras.py`. Leave as default to reproduce the training strategy used in the paper.
  * PyTorch models are configured in `config_pytorch.py`. Leave as default to reproduce the training strategy used in the paper.
* Produce results for the two test sets from `get_results_multiclass()`. By default, windows are handled appropriately to produce outputs over the same temporal window of 1.92 seconds for fair comparison. The different evaluation pipelines are handled in `runKeras.py` and `runPyTorchMulticlass.py`. ROC plots and confusion matrices are saved to disk on execution.
* (Optional) configure data input and output directories in `config.py`. 
* (Optional) configure or modify the feature creation in `feat_vggish.py`, `vggish_input.py` for Feat. A, and in `config.py` for Feat. B.

## Model configuration
The hyperparameter settings used for MSC are:
* `config_keras`: `epochs=80`, `batch_size=128`, `lengthscale=0.01`, `dropout=0.2`
* `config_pytorch`: `epochs=120`, `batch_size=32` or `batch_size=128` if the learning process is too slow, `lr=0.0003`, `dropout=0.05`, `pretrained=True`
