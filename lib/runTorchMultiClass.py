from ResNetDropoutSource import resnet50dropout, resnet18
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import config
import config_pytorch
from datetime import datetime
import os


class ResnetDropoutFull(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        n_output = 8
#     def __init__(self):
        super(ResnetDropoutFull, self).__init__()
        self.resnet = resnet50dropout(pretrained=config_pytorch.pretrained, dropout_p=0.2)
        # self.resnet = resnet18(pretrained=config_pytorch.pretrained)
        self.dropout = dropout
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # Figure out how to pass as parameter n_classes consistently: 1 with BCE loss, 2 with XENT loss? 8 for multiclass.
        self.fc1 = nn.Linear(2048,n_classes)  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer
#         self.apply(_weights_init)
    def forward(self, x):      
        x = self.resnet(x).squeeze()
#         x = self.fc1(x)
        # print(x.shape)
        x = self.fc1(F.dropout(x, p=self.dropout))
        # x = torch.sigmoid(x)
        return x


def build_dataloader(x_train, y_train, x_val=None, y_val=None, shuffle=True):
    x_train = torch.tensor(x_train).float()
    # model doesn't like this - removing
    # x_train = x_train.repeat(1,3,1,1)  # Repeat across 3 channels to match model expectation
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle)
    
    if x_val is not None:
        x_val = torch.tensor(x_val).float()
        x_val = x_val.repeat(1,3,1,1)
        print('just done x_val.repeat(1,3,1,1), might give a tensor shape error')
        y_val = torch.tensor(y_val,dtype=torch.long)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle)

        return train_loader, val_loader
    return train_loader


def train_model(x_train, y_train, class_weight=None, x_val=None, y_val=None,
                model = ResnetDropoutFull(config_pytorch.n_classes)):
    if x_val is not None:  # TODO: check dimensions when supplying validation data.
        train_loader, val_loader = build_dataloader(x_train, y_train, x_val, y_val)
    
    else:
        train_loader = build_dataloader(x_train, y_train)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model = model.to(device)

#     criterion = nn.BCELoss()  # Let's use CE loss for multiclass compatibility
    if class_weight is not None:
        print('Applying class weights:', class_weight)
        class_weight = torch.tensor([class_weight]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight) # Check if this loads correctly with weights=None
    optimiser = optim.Adam(model.parameters(), lr=config_pytorch.lr)

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    best_val_loss = np.inf
    best_val_acc = -np.inf

    # best_train_loss = np.inf
    best_train_acc = -np.inf

    best_epoch = -1
    checkpoint_name = None
    overrun_counter = 0

    for e in range(config_pytorch.epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in enumerate(train_loader):
#             print(np.shape(inputs))
#             print(np.shape(inputs[0]))
#             print(np.shape(inputs[1]), inputs[1])
#             x = inputs[0].to(device).detach()       
#             print('x shape', np.shape(x))
#             print('y before view call', np.shape(inputs[1].to(device).detach()))
#             y = inputs[1].to(device).detach().view(-1,1)


            ##Necessary in order to handle single and multi input feature spaces
            x = [xi.to(device).detach() for xi in inputs[:-1]]
            y = inputs[-1].to(device).detach().view(-1,1)
            if len(x) == 1:
                x = x[0]
#             print('shape x', np.shape(x))
            optimiser.zero_grad()
            y_pred = model(x)
#             print(np.shape(y))
#             print(np.shape(y_pred))
#             print('squeezed y_pred, y',np.shape(y_pred.squeeze()), np.shape(y.squeeze()))
            # print('y_pred: ', y_pred[0], '\ny: ', y.squeeze()[0])
            loss = criterion(y_pred, y.squeeze()) # will need to check for two-class also.
            # print('loss: ', loss)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())


            del x
            del y

        all_train_loss.append(train_loss/len(train_loader))

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        print('all_y', 'all_y_pred', np.shape(all_y.numpy()), np.shape(np.argmax(all_y_pred.numpy(), axis=1)))
        train_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))
        all_train_acc.append(train_acc)


        # Can add more conditions to support loss instead of accuracy. Use *-1 for loss inequality instead of acc
        if x_val is not None:
            val_loss, val_acc = test_model(model, val_loader, criterion, 0.5, device=device)  # This might not work multi.c.
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            acc_metric = val_acc
            best_acc_metric = best_val_acc
        else:
            acc_metric = train_acc
            best_acc_metric = best_train_acc
        if acc_metric > best_acc_metric:  
            # if checkpoint_name is not None:
                # os.path.join(os.path.pardir, 'models', 'pytorch', checkpoint_name)

            checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'

            torch.save(model.state_dict(), os.path.join(config.model_dir, 'pytorch', checkpoint_name))
            print('Saving model to:', os.path.join(config.model_dir, 'pytorch', checkpoint_name)) 
            best_epoch = e
            best_train_acc = train_acc
            best_train_loss = train_loss
            if x_val is not None:
                best_val_acc = val_acc
                best_val_loss = val_loss
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, Val Acc: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_acc, val_loss/len(val_loader), val_acc,  overrun_counter))
        else:
            print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_acc, overrun_counter))
        if overrun_counter > config_pytorch.max_overrun:
            break
    return model



def test_model(model, test_loader, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        test_loss = 0.0
        model.eval()
        
        all_y = []
        all_y_pred = []
        counter = 1
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x).squeeze()
            
            loss = criterion(y_pred, y)

            test_loss += loss.item()
            
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())
            
            del x
            del y
            del y_pred
            
            counter +=1

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        
        test_loss = test_loss/len(test_loader)
        test_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))



def evaluate_model(model, X_test, y_test, n_samples):
    # Determine number of classes: warning potential issue if predicted classes dont match
    # number of classes in y_test
    n_classes = len(np.unique(y_test)) 
                    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on {device}')

    x_test = torch.tensor(X_test).float()
    # vggish doesn't like this
    # x_test = x_test.repeat(1,3,1,1)

    y_test = torch.tensor(y_test).float()
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    y_preds_all = np.zeros([n_samples, len(y_test), n_classes])
    model.eval() # Important to not leak info from batch norm layers and cause other issues

    for n in range(n_samples):
        all_y_pred = []
        all_y = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x).squeeze()
            all_y.append(y.cpu().detach())

            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

        all_y_pred = torch.cat(all_y_pred)
        all_y = torch.cat(all_y)

        y_preds_all[n] = np.array(all_y_pred)
       
        test_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))
        print(test_acc)
    return y_preds_all


def load_model(filepath):
    # Instantiate model to inspect
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
    print(f'Training on {device}')
        
    model = ResnetDropoutFull(config_pytorch.n_classes)
    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # Load trained parameters from checkpoint (may need to download from S3 first)


    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint = model.load_state_dict(torch.load(filepath))

    return model