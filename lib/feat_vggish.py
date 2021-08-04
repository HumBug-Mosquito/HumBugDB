from PyTorch.vggish.vggish_input import waveform_to_examples
import torch
import librosa
import os
import skimage.util
import numpy as np
import config
import pickle
import math
# Sound type unique values: {'background', 'mosquito', 'audio'}; class labels: {1, 0, 0}

# Extract features from wave files with id corresponding to dataframe data_df.

def get_feat(data_df, data_dir, rate, min_duration):
    ''' Returns features extracted with Librosa. A list of features, with the number of items equal to the number of input recordings'''
    X = []
    y = []
    bugs = []
    skipped_files = []
    for row_idx_series in data_df.iterrows():
        row = row_idx_series[1]
        label_duration = row['length']
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row['name'])
            filename = os.path.join(data_dir, str(row['id']) + file_format)
            length = librosa.get_duration(filename = filename)
#             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)
            
            if math.isclose(length,label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                sig = waveform_to_examples(signal, rate)
                X.append(sig.unsqueeze(0))
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                elif row['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                print("File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['name'], label_duration, length))
                bugs.append([row['name'], label_duration, length])
                
        else:
            skipped_files.append([row['id'], row['name'], label_duration])
    return X, y, skipped_files, bugs



def get_signal(data_df, data_dir, rate, min_duration):
    ''' Returns raw audio with Librosa, and corresponding label longer than min_duration '''
    X = []
    y = []
    idx = 0
    bugs = []
    skipped_files = []
    label_dict = {}
    for row_idx_series in data_df.iterrows():
        row = row_idx_series[1]
        label_duration = row['length']
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row['name'])
            filename = os.path.join(data_dir, str(row['id']) + file_format)

            length = librosa.get_duration(filename = filename)
#             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)
            
            if math.isclose(length,label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                label_dict[idx] = [row['id'], row['name'],row['length']]
                idx+=1
                X.append(signal)
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                elif row['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                print("File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['name'], label_duration, length))
                bugs.append([row['name'], label_duration, length])
                
        else:
            skipped_files.append([row['id'], row['name'], label_duration])
    return X, y, label_dict, skipped_files, bugs



def reshape_feat(feats, labels):
    y_full = []
    for idx, feat in enumerate(feats):
        y_full.append(np.repeat(labels[idx], np.shape(feat)[1]))
    y = np.concatenate(y_full)

    x = torch.cat(feats, 1)
    x = x.squeeze(0)

    y = torch.tensor(y).float() 
    return x, y



def get_train_test_from_df(df_train, df_test_A, df_test_B, debug=False):
    
    pickle_name_train = 'vggish_feat_train.pickle'
     # step = window for test (no augmentation of test):
    pickle_name_test = 'vggish_feat_test.pickle'
    
    if not os.path.isfile(os.path.join(config.dir_out, pickle_name_train)):
        print('Extracting training features...')
        X_train, y_train, skipped_files_train, bugs_train = get_feat(data_df=df_train, data_dir = config.data_dir,
                                                                     rate=config.rate, min_duration=config.min_duration)
        X_train, y_train = reshape_feat(X_train, y_train)

        vggish_feat_train = {'X_train':X_train, 'y_train':y_train, 'bugs_train':bugs_train}

        if debug:
            print('Bugs train', bugs_train)
        
        with open(os.path.join(config.dir_out, pickle_name_train), 'wb') as f:
            pickle.dump(vggish_feat_train, f, protocol=4)
            print('Saved features to:', os.path.join(config.dir_out, pickle_name_train))

    else:
        print('Loading training features found at:', os.path.join(config.dir_out, pickle_name_train))
        with open(os.path.join(config.dir_out, pickle_name_train), 'rb') as input_file:
            vggish_feat = pickle.load(input_file)
            X_train = vggish_feat['X_train']
            y_train = vggish_feat['y_train']

    if not os.path.isfile(os.path.join(config.dir_out, pickle_name_test)):
        print('Extracting test features...')

        X_test_A, y_test_A, skipped_files_test_A, bugs_test_A = get_feat(data_df= df_test_A, data_dir = config.data_dir,
                                                                         rate=config.rate, min_duration=config.min_duration)
        X_test_B, y_test_B, skipped_files_test_B, bugs_test_B = get_feat(data_df= df_test_B, data_dir = config.data_dir,
                                                                         rate=config.rate, min_duration=config.min_duration)
        X_test_A, y_test_A = reshape_feat(X_test_A, y_test_A)  # Test should be strided with step = window.
        X_test_B, y_test_B = reshape_feat(X_test_B, y_test_B)  
        
        vggish_feat_test = {'X_test_A':X_test_A, 'X_test_B':X_test_B, 'y_test_A':y_test_A, 'y_test_B':y_test_B}

        if debug:
            print('Bugs test A', bugs_test_A)
            print('Bugs test B', bugs_test_B)

        
        with open(os.path.join(config.dir_out, pickle_name_test), 'wb') as f:
            pickle.dump(vggish_feat_test, f, protocol=4)
            print('Saved features to:', os.path.join(config.dir_out, pickle_name_test))
    else:
        print('Loading test features found at:', os.path.join(config.dir_out, pickle_name_test))
        with open(os.path.join(config.dir_out, pickle_name_test), 'rb') as input_file:
            vggish_feat = pickle.load(input_file)

            X_test_A = vggish_feat['X_test_A']
            y_test_A = vggish_feat['y_test_A']
            X_test_B = vggish_feat['X_test_B']
            y_test_B = vggish_feat['y_test_B']


    return X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B




def get_test_from_df(df_test_A, df_test_B, debug=False):
    

    pickle_name_test = 'vggish_feat_test.pickle'
    
    
    if not os.path.isfile(os.path.join(config.dir_out, pickle_name_test)):
        print('Extracting test features...')

        X_test_A, y_test_A, skipped_files_test_A, bugs_test_A = get_feat(data_df= df_test_A, data_dir = config.data_dir,
                                                                         rate=config.rate, min_duration=config.min_duration,
                                                                         n_feat=config.n_feat)
        X_test_B, y_test_B, skipped_files_test_B, bugs_test_B = get_feat(data_df= df_test_B, data_dir = config.data_dir,
                                                                         rate=config.rate, min_duration=config.min_duration,
                                                                         n_feat=config.n_feat)
        X_test_A, y_test_A = reshape_feat(X_test_A, y_test_A)  # Test should be strided with step = window.
        X_test_B, y_test_B = reshape_feat(X_test_B, y_test_B)  
        
        vggish_feat_test = {'X_test_A':X_test_A, 'X_test_B':X_test_B, 'y_test_A':y_test_A, 'y_test_B':y_test_B}

        if debug:
            print('Bugs test A', bugs_test_A)
            print('Bugs test B', bugs_test_B)

        
        with open(os.path.join(config.dir_out, pickle_name_test), 'wb') as f:
            pickle.dump(vggish_feat_test, f)
            print('Saved features to:', os.path.join(config.dir_out, pickle_name_test))
    else:
        print('Loading test features found at:', os.path.join(config.dir_out, pickle_name_test))
        with open(os.path.join(config.dir_out, pickle_name_test), 'rb') as input_file:
            vggish_feat = pickle.load(input_file)

            X_test_A = vggish_feat['X_test_A']
            y_test_A = vggish_feat['y_test_A']
            X_test_B = vggish_feat['X_test_B']
            y_test_B = vggish_feat['y_test_B']


    return X_test_A, y_test_A, X_test_B, y_test_B