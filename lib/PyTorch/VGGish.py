import torch.nn as nn
import torch
from torchvision.models import resnet50
from vggish.vggish import VGGish
from ResNetSource import resnet50dropout
import config
import torch.nn.functional as F # For dropout
import torch.nn.init as init
# Weight initialisation for dropout
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        print('Initialising weights')
        init.kaiming_normal_(m.weight)
class CoughModelVGGishVGGish(nn.Module):
    def __init__(self, preprocess=False):
        super(CoughModelVGGishVGGish, self).__init__()
        self.model_urls = config.vggish_model_urls
        self.vggish = VGGish(self.model_urls, postprocess=False, preprocess=preprocess)
        self.fc1 = nn.Linear(128, 1)
    def forward(self, x, fs=None):
        n_segments = x.shape[1]
        ##(Batch, Segments, C, H, W) -> (Batch*Segments, C, H, W)
        x = x.view(-1, 1, 96, 64)
        x = self.vggish.forward(x, fs)
        ##(Batch*Segments, Embedding) -> (Batch, Segments, Embedding)
        x = x.view(-1, n_segments, 128)
        ##TODO: Better solution than mean (RNN?)
        x = torch.mean(x, axis=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
class CoughModelMFCCResnet(nn.Module):
    def __init__(self):
        super(CoughModelMFCCResnet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)
    def forward(self, x): 
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
class CoughModelResnetDropout(nn.Module):
    def __init__(self, dropout=0.2):
#     def __init__(self):
        super(CoughModelResnetDropout, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.dropout = dropout
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)
#         self.apply(_weights_init)
    def forward(self, x):      
        x = self.resnet(x).squeeze()
#         x = self.fc1(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x
class CoughModelResnetDropoutFull(nn.Module):
    def __init__(self, dropout=0.2):
#     def __init__(self):
        super(CoughModelResnetDropoutFull, self).__init__()
        self.resnet = resnet50dropout(pretrained=True, dropout_p=0.2)
        self.dropout = dropout
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)
#         self.apply(_weights_init)
    def forward(self, x):      
        x = self.resnet(x).squeeze()
#         x = self.fc1(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x
class CoughModelVGGishResnet(nn.Module):
    def __init__(self):
        super(CoughModelVGGishResnet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)
    def forward(self, x):
        n_segments = x.shape[1]
        ##(Batch, Segments, C, H, W) -> (Batch*Segments, C, H, W)
        x = x.view(-1, 1, 96, 64)
        ##Resnet requires 3 input channels
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet.forward(x)
        ##(Batch*Segments, Embedding) -> (Batch, Segments, Embedding)
        x = x.view(-1, n_segments, 2048)
        ##TODO: Better solution than mean (RNN?)
        x = torch.mean(x, axis=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
class CoughModelCambridgeLinear(nn.Module):
    def __init__(self):
        super(CoughModelCambridgeLinear, self).__init__()
        self.fc1 = nn.Linear(477, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
class CoughModelCambridgeVGGishResnet(nn.Module):
    def __init__(self):
        super(CoughModelCambridgeVGGishResnet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048+477, 1)
    def forward(self, x_a):
        xv = x_a[0]
        x = x_a[1]
        n_segments = x.shape[1]
        ##(Batch, Segments, C, H, W) -> (Batch*Segments, C, H, W)
        x = x.view(-1, 1, 96, 64)
        ##Resnet requires 3 input channels
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet.forward(x)
        ##(Batch*Segments, Embedding) -> (Batch, Segments, Embedding)
        x = x.view(-1, n_segments, 2048)
        ##TODO: Better solution than mean (RNN?)
        x = torch.mean(x, axis=1)
        x = torch.cat([x, xv], dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x