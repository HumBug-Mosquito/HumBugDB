import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import sys
import os
import config



def get_results(y_pred_list, y_test, filename=None, show_plot_PE_MI=True, show_plot_roc=True, show_plot_cm=True, show_plot_pr=True):
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
        

        roc_score = sklearn.metrics.roc_auc_score(y_test, np.mean(out, axis=0)[:,1])
        print("mean ROC AUC:", roc_score)

        plot_roc("Test performance", y_test, np.mean(out, axis=0)[:,1], roc_score, filename, linestyle='--')

        auc_list = []
        for y in y_pred_list:
            auc_list.append(sklearn.metrics.roc_auc_score(y_test, y[:,1]))

        print("std ROC AUC:", np.std(auc_list))


    if show_plot_pr:
        plot_pr("Test performance", y_test, np.mean(out, axis=0)[:,1], filename)

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



def plot_roc(name, labels, predictions, roc_score, filename, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.figure(figsize=(4,4))
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.title(str(roc_score))
#     plt.xlim([-0.5,20])
#     plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(os.path.join(config.plot_dir, filename + '_ROC.pdf' ),bbox_inches='tight')
    plt.show()
    

def plot_pr(name, labels, predictions, filename):
    # Plot precision-recall curves
    
    area = average_precision_score(labels, predictions)
    print('PR-AUC: ', area)
    precision, recall, _ = precision_recall_curve(labels, predictions)
    plt.plot(recall, precision)
    plt.title('AUC={0:0.4f}'.format(area))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(config.plot_dir, filename + '_PR.pdf' ),bbox_inches='tight')
    plt.show()
   

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



def get_results_multiclass(y_test_CNN, y_pred_CNN, filename, classes):
  # First plot the default confusion matrix and save to text file.
    with open(os.path.join(config.plot_dir, filename + '_cm.txt' ), "w") as text_file:
        print(classification_report(y_test_CNN, np.argmax(y_pred_CNN, axis=1)), file=text_file)
    
    # Now plot multi-class ROC:
    compute_plot_roc_multiclass(y_test_CNN, y_pred_CNN, filename, classes, title=None)

    # Plot also precision-recall curves:
    compute_plot_pr_multiclass(y_test_CNN, y_pred_CNN, filename, classes, title=None)
    
    # Calculate confusion matrix
    cnf_matrix_unnorm = confusion_matrix(y_test_CNN, np.argmax(y_pred_CNN, axis=1))

    # Now normalise 
    cnf_matrix = cnf_matrix_unnorm/cnf_matrix_unnorm.sum(1)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(cnf_matrix, cmap=plt.cm.Blues) #plot confusion matrix grid
    threshold = cnf_matrix.max() / 2 #threshold to define text color
    for i in range(cnf_matrix.shape[0]): #print text in grid
        for j in range(cnf_matrix.shape[1]): 
            plt.text(j-0.2, i, cnf_matrix_unnorm[i,j], color="w" if cnf_matrix[i,j] > threshold else 'black')
    tick_marks = np.arange(len(classes)) #define labeling spacing based on number of classes
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    # plt.title(')
    plt.xlabel('Predicted label')
#   plt.colorbar(label='Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, filename + '_MSC_cm.pdf' ),bbox_inches='tight')
  
    return fig


def compute_plot_roc_multiclass(y_true, y_pred_prob, filename, classes, title=None):
    '''y_true: non-categorical y label. y_pred_prob: model.predict output of NN. '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(to_categorical(y_true)[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(to_categorical(y_true).ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    with open(os.path.join(config.plot_dir, filename + '_roc.txt' ), "w") as text_file:
        print(roc_auc, file=text_file)
    lw=2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='{0} (area = {1:0.3f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.plot_dir, filename + '_MSC_ROC.pdf' ),bbox_inches='tight')
    plt.show()




def compute_plot_pr_multiclass(y_true, y_pred_prob, filename, classes, title=None):
    # For each class
    n_classes = 8
    precision = dict()
    recall = dict()
    average_precision = dict()
    Y_test = to_categorical(y_true)
    y_score = y_pred_prob
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    

    with open(os.path.join(config.plot_dir, filename + '_pr.txt' ), "w") as text_file:
        print(average_precision, file=text_file)

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average (area = {0:0.3f})'
                  ''.format(average_precision["micro"]))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.3f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(os.path.join(config.plot_dir, filename + '_MSC_PR.pdf' ),bbox_inches='tight')






    # Tools for reshaping data:

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
      y: Array-like with class values to be converted into a matrix
          (integers from 0 to `num_classes - 1`).
      num_classes: Total number of classes. If `None`, this would be inferred
        as `max(y) + 1`.
      dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
      A binary matrix representation of the input. The class axis is placed
      last.
    Example:
    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> a = tf.constant(a, shape=[4, 4])
    >>> print(a)
    tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]
    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# For multi-class evaluation for both PyTorch and Keras: -> Used for Keras with evaluate model below
def evaluate_model_aggregated(model, X_test, y_test, n_samples):
    n_classes = 8
    preds_aggregated_by_mean = []
    y_aggregated_prediction_by_mean = []
    y_target_aggregated = []
    
    for idx, recording in enumerate(X_test):
        n_target_windows = len(recording)//2  # Calculate expected length: discard edge
        y_target = np.repeat(y_test[idx],n_target_windows) # Create y array of correct length
        preds = evaluate_model(model, recording, np.repeat(y_test[idx],len(recording)),n_samples) # Sample BNN
        preds = np.mean(preds, axis=0) # Average across BNN samples
        preds = preds[:n_target_windows*2,:] # Discard edge case
        preds = np.mean(preds.reshape(-1,2,n_classes), axis=1) # Average every 2 elements, across n_classes
        preds_y = np.argmax(preds, axis=1)  # Append argmax prediction (label output)
        y_aggregated_prediction_by_mean.append(preds_y)
        preds_aggregated_by_mean.append(preds)  # Append prob (or log-prob/other space)
        y_target_aggregated.append(y_target)  # Append y_target
    return np.concatenate(preds_aggregated_by_mean), np.concatenate(y_aggregated_prediction_by_mean), np.concatenate(y_target_aggregated)

# Helper function to run evaluate_model_aggregated for Keras models

def evaluate_model(model, X_test, y_test, n_samples):
    all_y_pred = []
    for n in range(n_samples):
        all_y_pred.append(model.predict(X_test))
    return all_y_pred