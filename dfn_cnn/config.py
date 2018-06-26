class LRConfig(object):
    """
    Holds logistic regression model hyperparams.
    
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float
    """
    def __init__(self,
                 height=45,
                 width=80,
                 channels=3,
                 classes=3,
                 batch_size=32,
                 epochs=3,
                 save_step=100,
                 learning_rate=0.01,
                 momentum=0.1):
        self.height = height
        self.width = width
        self.classes = classes
        self.channels = channels
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.momentum = momentum
        

    def __str__(self):
        """
        Get all attributs values.
        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "classes = {}\n".format(self.classes)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        return status


class DFNConfig(LRConfig):
    """
    Holds DFN model hyperparams.

    :param architecture: network dense architecture
    :type architecture: list of int
    :param activations: list of different activation functions
    :type actvations: list of str, None
    """
    def __init__(self, architecture=[100, 3], activations=None, input_dim=None):
        super(DFNConfig, self).__init__()
        self.architecture = architecture
        if activations is None:
            self.activations = ["relu"] * (len(architecture) - 1) +  ["softmax"]
        else:
            self.activations = activations
        if input_dim is None:
            self.input_dim = self.height * self.width * self.channels
        else:
            self.input_dim = input_dim
            

    def __str__(self):
        """
        Get all attributs values.

        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "input_dim = {}\n".format(self.input_dim)
        status += "architecture = {}\n".format(self.architecture)
        status += "activations = {}\n".format(self.activations)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        return status


class CNNConfig(LRConfig):
    """
    Holds CNN model hyperparams.

    :param conv_architecture: convolutional architecture
    :type conv_architecture: list of int
    :param kernel_sizes: filter sizes
    :type kernel_sizes: list of int
    :param pool_kernel: pooling filter sizes
    :type pool_kernel: list of int
    """
    def __init__(self,
                 architecture=[100, 3],
                 activations = None,
                 conv_architecture=[32, 64],
                 conv_activations = None,
                 kernel_sizes=None,
                 pool_kernel=None):
        super(CNNConfig, self).__init__()
        self.architecture = architecture
        if activations is None:
            self.activations = ["relu"] * (len(architecture) - 1) +  ["softmax"]
        else:
            self.activations = activations
        self.conv_architecture = conv_architecture
        if kernel_sizes is None:
            self.kernel_sizes = [5] * len(self.conv_architecture)
        else:
            self.kernel_sizes = kernel_sizes
        if pool_kernel is None:
            self.pool_kernel = [2] * len(self.conv_architecture)
        else:
            self.pool_kernel = pool_kernel
        if conv_activations is None:
            self.conv_activations = ["relu"] * len(self.conv_architecture)
        else:
            self.conv_activations = conv_activations
            

    def __str__(self):
        """
        Get all attributs values.

        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "architecture = {}\n".format(self.architecture)
        status += "activations = {}\n".format(self.activations)
        status += "conv_activations = {}\n".format(self.conv_activations)
        status += "conv_architecture = {}\n".format(self.conv_architecture)
        status += "kernel_sizes = {}\n".format(self.kernel_sizes)
        status += "pool_kernel = {}\n".format(self.pool_kernel)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        return status