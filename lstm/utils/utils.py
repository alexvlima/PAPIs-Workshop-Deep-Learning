import itertools

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def plot_train_validation_graph(train_values,
                                validation_values,
                                legend_loc='lower right'):
    """
    This function plots the train and validation comparison graph
    We can plot the difference between loss or accuracy between train and
    validation set.

    :param train_values: a list of train losses or accuracy
    :type train_values: list
    :param validation_values: a list of validation losses or accuracy
    :type validation_values: list
    :param legend_loc: location of the legend in the graph
    :type legend_loc: str
    """
    plt.plot(train_values)
    plt.plot(validation_values)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=legend_loc)
    plt.show()


def plot_confusion_matrix(truth,
                          predictions,
                          classes,
                          normalize=False,
                          save=False,
                          cmap=plt.cm.Oranges):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    'cmap' controls the color plot. colors:
    https://matplotlib.org/1.3.1/examples/color/colormaps_reference.html
    :param truth: true labels
    :type truth: np array
    :param predictions: model predictions
    :type predictions: np array
    :param classes: list of classes in order
    :type classes: list
    :param normalize: param to normalize cm matrix
    :type normalize: bool
    :param save: param to save cm plot
    :type save: bool
    :param cmap: plt color map
    :type cmap: plt.cm
    """
    acc = np.array(truth) == np.array(predictions)
    size = float(acc.shape[0])
    acc = np.sum(acc.astype("int32")) / size
    title = "Confusion matrix of {0} examples\n accuracy = {1:.6f}".format(int(size),  # noqa
                                                                           acc)
    cm = confusion_matrix(truth, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.show()
