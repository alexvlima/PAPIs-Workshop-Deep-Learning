import re

import matplotlib.pyplot as plt
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class TweetDataset:
    """
    Class used to handle the GOP twitted dataset.
    """

    def __init__(self, path, num_words):
        """
        # Arguments

            path: path where the data is located.
            num_words: max number of words included. The ranking of the words will
                    be based on the frequency of the word all of the tweets.

        """
        self.path = path
        self.num_words = num_words

        self.load_dataset()

    def plot_dataset_info(self):
        plt.gcf().subplots_adjust(bottom=0.15)

        all_sentiment = self.dataset.groupby(['sentiment']).size()
        sentiment_plot = all_sentiment.plot.bar()
        ax = sentiment_plot.get_figure()

        ax.savefig('images/tweet.png')
        

    def preprocess_text(self):
        self.dataset['text'] = self.dataset['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        self.dataset['text'] = self.dataset['text'].apply(lambda x: re.sub(r'RT', '', x))

    def create_tokenizer(self, texts):
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(texts)

        return tokenizer

    def format_text(self, tokenizer, data):

        X = tokenizer.texts_to_sequences(data['text'].values)
        X = pad_sequences(X)
        y = pd.get_dummies(data['sentiment']).values

        return X, y

    def load_dataset(self):
        dataset = pd.read_csv(self.path)

        self.dataset = dataset[['text', 'sentiment']]
        # self.preprocess_text()
        train_x = self.dataset.text.values
        train_y = self.dataset.sentiment.values

        #train_x, validation_x, train_y, validation_y = train_test_split(
        #    X, y,
        #    test_size=0.1,
        #    random_state=42,
        #    stratify=y)

        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y,
            test_size=0.1,
            random_state=42,
            stratify=train_y)

        tokenizer = self.create_tokenizer(train_x)
        train_data = pd.DataFrame({'text': train_x, 'sentiment': train_y}) 
        train_data = train_data[['text', 'sentiment']]
        #val_data = pd.DataFrame({'text': validation_x, 'sentiment': validation_y})
        test_data = pd.DataFrame({'text': test_x, 'sentiment': test_y})
        test_data = test_data[['text', 'sentiment']]

        train_data.to_csv('data/gop_tweets/train.csv')
        #val_data.to_csv('data/val.csv')
        test_data.to_csv('data/gop_tweets/test.csv')
    
        train_x, train_y = self.format_text(tokenizer, train_data)
        #val_x, val_y = self.format_text(tokenizer, val_data)
        test_x, test_y = self.format_text(tokenizer, test_data)
        
        self.train_data = (train_x, train_y)
        #self.val_data = (val_x, val_y)
        self.test_data = (test_x, test_y)
