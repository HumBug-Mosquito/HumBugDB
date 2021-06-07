import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
import os
import config

def get_results(y_pred_list, y_test, filename=None, show_plot_PE_MI=True, show_plot_roc=True, show_plot_cm=True):
    ''' Input: prediction list from model, y_test. y_test is a 1D Torch array (or 1D numpy for Keras).'''


    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)

    out = y_pred_list # out: output
    G_X, U_X, log_prob = active_BALD(np.log(out), y_test, 2)


    if show_plot_PE_MI:
        start_vis = 0
        end_vis = len(y_test)
        plt.figure(figsize=(12,5))
        plt.title('Mean pred, pred_entropy, and MI for samples {from, to}: ' + str(start_vis) + ', ' + str(end_vis))
        plt.plot(np.arange(start_vis, end_vis), np.mean(out,axis=0)[start_vis:end_vis,1], 'ko', label='$\hat{y}_{mean}$')
        plt.plot(np.arange(start_vis, end_vis), y_test[start_vis:end_vis], 'r--', label='${y}_{test}$')
        plt.plot(np.arange(start_vis, end_vis),G_X[start_vis:end_vis], label='pred_entropy')
        plt.plot(np.arange(start_vis, end_vis),U_X[start_vis:end_vis], label='MI')
        plt.xlabel('Feature window')
        plt.legend()
        plt.savefig(os.path.join(config.plot_dir, filename + '_PE_MI.pdf' ),bbox_inches='tight')
        plt.show()


    if show_plot_roc:
        plot_roc("Test performance", y_test, np.mean(out, axis=0)[:,1], filename, linestyle='--')
        print("mean ROC AUC:", sklearn.metrics.roc_auc_score(y_test, np.mean(out, axis=0)[:,1]))

        auc_list = []
        for y in y_pred_list:
            auc_list.append(sklearn.metrics.roc_auc_score(y_test, y[:,1]))

        print("std ROC AUC:", np.std(auc_list))

    if show_plot_cm:
        # Calculate confusion matricies
        cm_list = []
        for i in np.arange(len(out)):
            cm_list.append(confusion_matrix(y_test, np.argmax(out[i],-1)))

        cm = []
        for item in cm_list:
            cm.append(item.astype('float') / item.sum(axis=1)[:, np.newaxis] *100)
        cm_mean = np.mean(cm, axis = 0) # Convert mean to normalised percentage
        cm_std = np.std(cm, axis = 0) # Standard deviation also in percentage


        np.set_printoptions(precision=4)




        class_names= np.array(['Noise', 'Mozz'])

        # Plot normalized confusion matrix
        plot_confusion_matrix(cm_mean,  std=cm_std, classes=class_names, filename=filename, normalize=False)
        # plt.tight_layout()
        # plt.savefig('Graphs/cm_RF_BNN.pdf', bbox_inches='tight')
        
        plt.show()
        
    return G_X, U_X, log_prob




def active_BALD(out, X, n_classes):

    log_prob = np.zeros((out.shape[0], X.shape[0], n_classes))
    score_All = np.zeros((X.shape[0], n_classes))
    All_Entropy = np.zeros((X.shape[0],))
    for d in range(out.shape[0]):
#         print ('Dropout Iteration', d)
#         params = unflatten(np.squeeze(out[d]),layer_sizes,nn_weight_index)
        log_prob[d] = out[d]
        soft_score = np.exp(log_prob[d])
        score_All = score_All + soft_score
        #computing F_X
        soft_score_log = np.log2(soft_score+10e-15)
        Entropy_Compute = - np.multiply(soft_score, soft_score_log)
        Entropy_Per_samp = np.sum(Entropy_Compute, axis=1)
        All_Entropy = All_Entropy + Entropy_Per_samp
 
    Avg_Pi = np.divide(score_All, out.shape[0])
    Log_Avg_Pi = np.log2(Avg_Pi+10e-15)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy, out.shape[0])
    F_X = Average_Entropy
    U_X = G_X - F_X
# G_X = predictive entropy
# U_X = MI
    return G_X, U_X, log_prob



def plot_roc(name, labels, predictions, filename, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.figure(figsize=(4,4))
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
#     plt.xlim([-0.5,20])
#     plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(os.path.join(config.plot_dir, filename + '_ROC.pdf' ),bbox_inches='tight')
    
    

def plot_confusion_matrix(cm, classes, std, filename=None,
                          normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


#     std = std * 100
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] *100
#         std = std.astype('float') / std.sum(axis=1)[:, np.newaxis] *100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, as input by user')

    print(cm)

    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,

           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.2f'
    fmt_std = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt) + '±' + format(std[i, j], fmt_std),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, filename + '_cm.pdf' ))
    return ax