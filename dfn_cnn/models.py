from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras import optimizers




class LogisticRegression(Sequential):
    """
    Logistic regression model.
    
    You may find nn.Linear and nn.Softmax useful here.
    
    :param config: hyper params configuration
    :type config: LRConfig
    """
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        input_dim = config.height * config.width * config.channels
        self.add(Dense(units=config.classes,
                       input_dim=input_dim, activation='softmax'))
        sgd = optimizers.SGD(lr=config.learning_rate,
                     momentum=config.momentum)
        self.compile(loss='categorical_crossentropy', 
                      optimizer=sgd,
                      metrics=['accuracy'])

class DFN(Sequential):
    """
    Deep Feedforward Network.
    
    The method self._modules is useful here.
    The class nn.ReLU() is useful too.

    :param config: hyper params configuration
    :type config: DFNConfig
    """
    def __init__(self, config):
        # YOUR CODE HERE:
        super(DFN, self).__init__()
        input_dim = config.input_dim
        architecture_size = len(config.architecture)
        size_arc = list(zip(config.architecture, config.activations))
        for i, tuple_size in enumerate(size_arc):
            size, activation = tuple_size
            if i == 0:
                self.add(Dense(units=size,
                               input_dim=input_dim, activation=activation))
            else:
                self.add(Dense(units=size,activation=activation))
        sgd = optimizers.SGD(lr=config.learning_rate,
             momentum=config.momentum)
        self.compile(loss='categorical_crossentropy', 
                      optimizer=sgd,
                      metrics=['accuracy'])
        # END YOUR CODE


class CNN(Sequential):
    """
    Convolutional Neural Network.
    
    The method self._modules is useful here.
    The class nn.ReLU() is useful too.

    :param config: hyper params configuration
    :type config: CNNConfig
    """
    def __init__(self, config):
        # YOUR CODE HERE:
        super(CNN, self).__init__()
        conv_architecture_size = len(config.conv_architecture)
        architecture_size = len(config.architecture)
        conv_arch = list(zip(config.conv_architecture, config.kernel_sizes, config.pool_kernel, config.conv_activations))
        size_arc = list(zip(config.architecture, config.activations))
        for i, quadruple in enumerate(conv_arch):
            filters, kernel_size, pool_size, activation = quadruple
            if i == 0:
                self.add(Conv2D(filters,
                                 (kernel_size, kernel_size),
                                 padding='same',
                                 activation=activation,
                                 input_shape=(config.height, config.width, config.channels)))
            else:
                self.add(Conv2D(filters,
                                 (kernel_size, kernel_size),
                                 padding='same',
                                 activation=activation))
                
            self.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        self.add(Flatten())
        for tuple_size in size_arc:
            size, activation = tuple_size
            self.add(Dense(units=size,activation=activation))
        sgd = optimizers.SGD(lr=config.learning_rate,
                             momentum=config.momentum)
        self.compile(loss='categorical_crossentropy', 
                      optimizer=sgd,
                      metrics=['accuracy'])
        # END YOUR CODE
  