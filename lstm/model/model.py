from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D


class LSTMConfig:

    def __init__(self,
                 batch_size=32,
                 embedding_size=50,
                 num_words=2000,
                 lstm_units=64,
                 num_classes=3,
                 num_epochs=10):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_words = num_words
        self.lstm_units=lstm_units
        self.num_classes = num_classes
        self.num_epochs = num_epochs

    def __str__(self):
        status = ''
        status += 'batch size: {}\n'.format(self.batch_size)
        status += 'embedding size: {}\n'.format(self.embedding_size)
        status += 'num words: {}\n'.format(self.num_words)
        status += 'lstm units: {}\n'.format(self.lstm_units)
        status += 'num classes: {}\n'.format(self.num_classes)
        status += 'num epochs: {}\n'.format(self.num_epochs)

        return status


class Model:

    def __init__(self, lstm_config):
        self.lstm_config = lstm_config

    def build_model(self):
        raise NotImplementedError

    def fit(self, train_data, val_data=None):
        train_x = train_data[0]
        train_y = train_data[1]

        validation_split = 0 if val_data else 0.2

        return self.model.fit(
             x=train_x,
             y=train_y,
             epochs=self.lstm_config.epochs,
             verbose=2,
             validation_split=validation_split,
             validation_data=val_data)
    
    def evaluate(self, data):
        data_x = data[0]
        data_y = data[1]

        loss, acc = self.model.evaluate(
            x=data_x,
            y=data_y,
            verbose=2,
            batch_size=self.lstm_config.batch_size)

        return loss, acc

    def predict(self, data):
        return self.model.predict(data, batch_size=self.lstm_config.batch_size, verbose=2)
