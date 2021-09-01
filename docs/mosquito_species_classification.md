# Task 2: Mosquito Species Classification

This task showcases an example use of the code for classification of mosquito species. Over 1,000 individually captured wild mosquitoes of 8 species are included in the dataset for this task. The data collection, splitting, model training and evaluation is given in  [*HumBugDB: A Large-scale Acoustic Mosquito
Dataset*
](https://openreview.net/pdf?id=vhjsBtq9OxO).

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

## Trained models
The results of the model training are included in this repository for the purpose of reproducibility, or if you would like to use standalone models for your own applications. Note also, that data was split into 75% training and 25% testing for the purpose of validation with 5 distinct random seeds. If you wish to use the model as a standalone classifier, or have your own test set for evaluation, it is worth re-training any model of your choice on the full dataset. Consider also that the input pipeline of audio and feature processing has to be matched identically for correct function. If you wish to use any of the trained models from the experiments reported in Section 5.2 of the paper, you may download any of the following models which are linked to [GitHub release v2.0](https://github.com/HumBug-Mosquito/HumBugDB/releases/tag/2.0):


| model | feat | model_class | random_seed | roc auc micro average | roc class 0 | roc class 1 | roc class 2 | roc class 3 | roc class 4 | roc class 5 | roc class 6 | roc class 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_e16_2021_08_06_16_02_47.pth | A | vgg | 5 | 0.932234912 | 0.865894952 | 0.83700881 | 0.981410161 | 0.95165334 | 0.860528766 | 0.940799337 | 0.851451541 | 0.888144934 |
| Win_30_Stride_5_2021_08_18_14_43_54-e80accuracy0.9556.hdf5 | A | mozzbnnv2 | 5 | 0.91904086 | 0.839702961 | 0.82806624 | 0.957000099 | 0.915464086 | 0.782110897 | 0.925072898 | 0.872454259 | 0.922532962 |
| model_e90_2021_08_07_00_03_43.pth | A | resnet18 | 5 | 0.931248958 | 0.863256557 | 0.860699366 | 0.984054368 | 0.926245087 | 0.887456797 | 0.932173608 | 0.939046254 | 0.8312084 |
| model_e94_2021_08_06_21_37_05.pth | A | resnet50 | 5 | 0.93253096 | 0.835154286 | 0.841876456 | 0.987189572 | 0.955384618 | 0.896955707 | 0.949010715 | 0.901273771 | 0.873272756 |
| 2021_08_17_15_13_22_model_e004.pth | B | vgg | 5 | 0.915 | 0.833 | 0.804 | 0.982 | 0.885 | 0.842 | 0.917 | 0.854 | 0.903 |
| Win_30_Stride_5_2021_07_31_03_25_35-e69accuracy0.9608.hdf5 | B | mozzbnnv2 | 5 | 0.929262474 | 0.865100434 | 0.881182953 | 0.977386229 | 0.903403219 | 0.83672269 | 0.909909302 | 0.85067915 | 0.915429688 |
| model_e77_2021_07_31_20_01_44.pth | B | resnet18 | 5 | 0.963325937 | 0.922952627 | 0.93688166 | 0.963522226 | 0.948526439 | 0.96839646 | 0.957300362 | 0.942416461 | 0.959894954 |
| model_e73_2021_08_01_18_19_10.pth | B | resnet50 | 5 | 0.905654818 | 0.845785976 | 0.84078992 | 0.928063949 | 0.869965807 | 0.838256959 | 0.856039843 | 0.800616813 | 0.75460568 |
| model_e31_2021_09_01_16_48_49.pth | A | vgg | 10 | 0.922214528 | 0.867614042 | 0.833689278 | 0.958738376 | 0.922244059 | 0.872235559 | 0.911611652 | 0.856992811 | 0.908273377 |
| Win_30_Stride_5_2021_08_18_15_04_54-e72accuracy0.9536.hdf5 | A | mozzbnnv2 | 10 | 0.915984533 | 0.847369672 | 0.803561727 | 0.93257473 | 0.915201796 | 0.771004405 | 0.923909622 | 0.875191371 | 0.903098825 |
| model_e97_2021_08_16_23_26_37.pth | A | resnet18 | 10 | 0.893936217 | 0.745813935 | 0.852874495 | 0.980367672 | 0.766323681 | 0.864905183 | 0.951193018 | 0.88917851 | 0.872056235 |
| model_e99_2021_08_16_21_43_19.pth | A | resnet50 | 10 | 0.866380037 | 0.664699416 | 0.856244822 | 0.99003914 | 0.926834309 | 0.921594794 | 0.932827319 | 0.915427542 | 0.935155059 |
| 2021_08_17_15_56_18_model_e015.pth | B | vgg | 10 | 0.921 | 0.862 | 0.835 | 0.969 | 0.922 | 0.869 | 0.86 | 0.805 | 0.897 |
| Win_30_Stride_5_2021_08_03_09_42_31-e60accuracy0.9614.hdf5 | B | mozzbnnv2 | 10 | 0.931859358 | 0.86845592 | 0.861570264 | 0.973078571 | 0.930856298 | 0.851296769 | 0.910464906 | 0.883630316 | 0.900182657 |
| model_e28_2021_08_02_20_16_25.pth | B | resnet18 | 10 | 0.895728599 | 0.78407814 | 0.842139939 | 0.956942965 | 0.879772135 | 0.853519557 | 0.901586647 | 0.854797686 | 0.909581471 |
| model_e17_2021_09_01_21_45_45.pth | B | resnet50 | 10 | 0.902792519 | 0.871300068 | 0.851406672 | 0.982231351 | 0.748782051 | 0.899344438 | 0.912752772 | 0.885534219 | 0.893951217 |
| 2021_08_17_11_40_29_model_e033.pth | A | vgg | 21 | 0.902 | 0.818 | 0.795 | 0.957 | 0.924 | 0.794 | 0.822 | 0.825 | 0.932 |
| Win_30_Stride_5_2021_09_01_20_12_59-e80accuracy0.9509.hdf5 | A | mozzbnnv2 | 21 | 0.899647172 | 0.817019655 | 0.811112742 | 0.947866753 | 0.913630081 | 0.780481435 | 0.866334947 | 0.734788778 | 0.887292769 |
| model_e97_2021_08_17_15_08_51.pth | A | resnet18 | 21 | 0.895742079 | 0.716786226 | 0.851926956 | 0.984885935 | 0.936569178 | 0.839070979 | 0.891654499 | 0.815610957 | 0.910371174 |
| model_e95_2021_08_17_17_46_53.pth | A | resnet50 | 21 | 0.897739842 | 0.756215318 | 0.866158382 | 0.991774875 | 0.962410988 | 0.828453424 | 0.918563488 | 0.761780577 | 0.932719666 |
| 2021_08_17_13_28_25_model_e009.pth | B | vgg | 21 | 0.901 | 0.84 | 0.792 | 0.972 | 0.919 | 0.83 | 0.825 | 0.634 | 0.8998 |
| Win_30_Stride_5_2021_07_31_02_31_05-e77accuracy0.9633.hdf5 | B | mozzbnnv2 | 21 | 0.911029574 | 0.850298187 | 0.849051739 | 0.946992923 | 0.932234413 | 0.814099196 | 0.8432176 | 0.726091404 | 0.887796084 |
| model_e78_2021_07_31_02_00_40.pth | B | resnet18 | 21 | 0.949005125 | 0.907429827 | 0.901866011 | 0.977285638 | 0.975732819 | 0.915732133 | 0.904345599 | 0.87113213 | 0.865580568 |
| model_e14_2021_08_03_11_51_30.pth | B | resnet50 | 21 | 0.877859133 | 0.820843273 | 0.811504664 | 0.975792967 | 0.794714671 | 0.856246662 | 0.837392094 | 0.679175551 | 0.830601496 |
| model_e26_2021_08_06_14_37_16.pth | A | vgg | 42 | 0.927631184 | 0.872607243 | 0.817859208 | 0.959279906 | 0.949717764 | 0.844207591 | 0.922656942 | 0.831691622 | 0.927516481 |
| Win_30_Stride_5_2021_08_18_14_10_40-e80accuracy0.9524.hdf5 | A | mozzbnnv2 | 42 | 0.9180643 | 0.841334122 | 0.804273168 | 0.950169146 | 0.925884304 | 0.80925792 | 0.913443796 | 0.866946067 | 0.938804335 |
| model_e90_2021_08_17_13_16_16.pth | A | resnet18 | 42 | 0.909380167 | 0.789496132 | 0.873975027 | 0.977699076 | 0.856404728 | 0.908926751 | 0.925815234 | 0.878187959 | 0.942115871 |
| model_e97_2021_08_17_03_47_25.pth | A | resnet50 | 42 | 0.906880521 | 0.749260015 | 0.873088771 | 0.978550184 | 0.956620033 | 0.917703604 | 0.952306869 | 0.888178063 | 0.957539805 |
| 2021_08_17_14_50_50_model_e009.pth | B | vgg | 42 | 0.917 | 0.845 | 0.82 | 0.926 | 0.919 | 0.897 | 0.821 | 0.806 | 0.92 |
| Win_30_Stride_5_2021_07_26_12_59_05-e80accuracy0.9635.hdf5 | B | mozzbnnv2 | 42 | 0.933 | 0.865 | 0.881 | 0.959 | 0.935 | 0.874 | 0.886 | 0.849 | 0.934 |
| model_e96_2021_07_30_21_07_26.pth | B | resnet18 | 42 | 0.91 | 0.846 | 0.843 | 0.937 | 0.928 | 0.825 | 0.762 | 0.796 | 0.833 |
| model_e91_2021_07_30_05_40_54.pth | B | resnet50 | 42 | 0.913 | 0.841 | 0.849 | 0.933 | 0.919 | 0.847 | 0.811 | 0.837 | 0.869 |
| 2021_08_17_22_48_52_model_e028.pth | A | vgg | 100 | 0.919 | 0.864 | 0.822 | 0.97 | 0.929 | 0.807 | 0.897 | 0.804 | 0.952 |
| Win_30_Stride_5_2021_08_18_18_31_04-e80accuracy0.9527.hdf5 | A | mozzbnnv2 | 100 | 0.916486854 | 0.838907897 | 0.823353413 | 0.954740721 | 0.910908735 | 0.791817177 | 0.907211171 | 0.799130658 | 0.939586442 |
| model_e94_2021_08_17_22_53_57.pth | A | resnet18 | 100 | 0.876234067 | 0.661858136 | 0.809590971 | 0.984120248 | 0.792436097 | 0.832788557 | 0.910659897 | 0.852697223 | 0.94061549 |
| model_e98_2021_08_18_01_51_12.pth | A | resnet50 | 100 | 0.86285817 | 0.681198779 | 0.83041756 | 0.978896694 | 0.906549307 | 0.895908658 | 0.907146523 | 0.897480182 | 0.903836002 |
| 2021_08_17_16_55_17_model_e011.pth | B | vgg | 100 | 0.913 | 0.819 | 0.811 | 0.97 | 0.9 | 0.888 | 0.86 | 0.757 | 0.946 |
| Win_30_Stride_5_2021_07_31_13_20_53-e68accuracy0.9601.hdf5 | B | mozzbnnv2 | 100 | 0.929102423 | 0.878397934 | 0.860689621 | 0.972328534 | 0.91635272 | 0.878429173 | 0.865371148 | 0.792804724 | 0.945413523 |
| model_e99_2021_07_31_16_58_14.pth | B | resnet18 | 100 | 0.914389731 | 0.84715172 | 0.835942849 | 0.960771422 | 0.904842329 | 0.863138386 | 0.804853134 | 0.84062826 | 0.885446576 |
| model_e85_2021_08_02_07_17_14.pth | B | resnet50 | 100 | 0.919107339 | 0.871398098 | 0.877041201 | 0.933460617 | 0.895721447 | 0.872414513 | 0.777828907 | 0.811950063 | 0.870453402 |
