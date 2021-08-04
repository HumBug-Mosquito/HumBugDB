# -*- coding: utf-8 -*-
"""pip install pad
Created on Mon Aug  2 14:41:39 2021

@author: benpg
"""

"""
Minimum working version not in a notebook
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../torchvggish/torchvggish'))
sys.path.append('../lib')

import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import config
from sklearn.utils import shuffle, class_weight
from tqdm import tqdm
import datetime
import pickle
from vggish_input import wavfile_to_examples
from os.path import join
from evaluate import get_results, plot_confusion_matrix_multiclass, compute_plot_roc_multiclass



model = torch.hub.load('harritaylor/torchvggish', 'vggish', pretrained=True)
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# file_dir = '../torchvggish'
# try: urllib.URLopener().retrieve(url, os.path.join(file_dir, filename))
# except: urllib.request.urlretrieve(url, os.path.join(file_dir, filename))

def load_vggish_feats(file, dir='../outputs/features/vggish'):
  if os.path.exists(join(dir, file)):
    with open(join(dir, file), 'rb') as f:
      [X, y] = pickle.load(f)
  return X, y

test_file = 'test_feats_vggish_20210802124327.pkl'
train_file = 'train_feats_vggish_20210802124327.pkl'
X_train, y_train = load_vggish_feats(train_file)
X_test, y_test = load_vggish_feats(test_file)

#%% MODEL
from runTorchMultiClass import train_model, test_model, load_model, evaluate_model

num_classes = 8 # in a config file or something
class VGGishMulticlass(nn.Module):
  def __init__(self, num_classes, vggish_model=None, dropout=0.2): # dropout?
    super(VGGishMulticlass, self).__init__()
    self.dropout = dropout
    # TODO go back to how it was before, so you can control preprocessing
    if vggish_model is not None:
      self.vggish_embedding = vggish_model
    else: 
      self.vggish_embedding = torch.hub.load('harritaylor/torchvggish', 'vggish', pretrained=True)
    self.fc1 = nn.Linear(128, num_classes)
  def forward(self, x):
      x = self.vggish_embedding(x).squeeze()
      x = self.fc1(F.dropout(x,p=self.dropout))
      # x = torch.sigmoid(x) # TODO: is this necessary?
      return x

model.preprocess = False
test_embeddings = model.forward(X_test[:10])
modelvsforwardtest = model(X_test[:10])
plt.plot(test_embeddings, 'o')

class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)


test_model = VGGishMulticlass(num_classes=8, vggish_model=model)
print(test_model.forward(X_test[:100]).shape)

trained_model = train_model(X_train, y_train, 
                            # class_weight=class_weights, 
                            model=VGGishMulticlass(num_classes, model))
