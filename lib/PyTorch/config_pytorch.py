dropout = 0.2

# Settings for binary classification for main.ipynb

lr = 0.0015
max_overrun = 30
epochs = 50
batch_size = 32
pretrained = True

# VGG-ish
vggish_model_urls = {'vggish': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth'}

# Settings for multi-class classification with 8 species for species_classification.ipynb
n_classes = 8
epochs = 100

