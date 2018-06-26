import numpy as np


def extend_dataset_flip_axis(data,
                             labels,
                             height=90,
                             width=160,
                             channels=3):
    """
    Balance and extend dataset
    by generating new images flipping the horizontal
    axis (only applicable to images labeled 'left' or 'right').
    This function is hard-coded, it assumes the following codification:
        - "forward": 0
        - "left": 1
        - "right": 2
    :param data: dataset
    :type data: np.array
    :param label: labels
    :type label: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: extended images, extended labels
    :rtype: np.array, np.array
    """
    all_images = []
    all_labels = []
    flat_shape = data.shape[1]
    for i in range(data.shape[0]):
        orig_label = labels[i]
        if orig_label == 0:
            continue
        frame = data[i].reshape((height, width, channels))

        if orig_label == 1:
            flip_cmd = 2
        else:
            flip_cmd = 1
        flip = np.flip(frame, axis=1)
        flip = np.array(flip.reshape(flat_shape))
        all_images.append(flip)
        all_labels.append(flip_cmd)
    all_labels = np.array(all_labels).astype('uint8')
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    extended_images = np.concatenate((data, all_images), axis=0)
    extended_labels = np.concatenate((labels, all_labels), axis=0)
    return extended_images, extended_labels


def labels2csv(labels, csv_path):
    """
    Transform an array of labels into a csv file
    to be submitted on the kaggle competition
    https://www.kaggle.com/c/mac0460-self-driving

    :param labels: labels
    :type labels: np.array
    :param csv_path: path to save csv file
    :type csv_path: str
    """
    with open(csv_path, "w") as file:
        file.write("id,label\n")
        for i, label in enumerate(labels):
            file.write("{},{}\n".format(i, label))


def randomize_in_place(list1, list2, init=0):
    """
    Function to randomize two lists in the same way.

    :param list1: list
    :type list1: list or np.array
    :param list2: list
    :type list2: list or np.array
    :param init: seed
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)


def generator2array(generator):
    """
    Extract data and labels from generatos

    :param generator: image generator
    :type generator: ImageDataGenerator
    :return: data, labels
    :rtype: np.array, np.array
    """
    data = []
    labels = []

    for i in range(len(generator)):
        images, label = generator.__getitem__(i)
        data.append(images)
        labels.append(label)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels
