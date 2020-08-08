import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from numpy import zeros
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config.config import Config

x = tf.compat.v1

x.disable_eager_execution()

keras = tf.keras

Sequential = tf.keras.Sequential

Embedding = tf.keras.layers.Embedding

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

Tokenizer = tf.keras.preprocessing.text.Tokenizer

LSTM = tf.keras.layers.LSTM

Bidirectional = tf.keras.layers.Bidirectional

Dense = keras.layers.Dense

Dropout = keras.layers.Dropout

Input = keras.layers.Input

Lambda = keras.layers.Lambda

Model = tf.keras.Model

SGD = tf.keras.optimizers.SGD

Adam = tf.keras.optimizers.Adam

EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

vocabulary = pd.read_csv(Config.vocabulary, dtype=object, encoding='utf-8').fillna('')
vocabulary = vocabulary[['article_sentence']]
vocabulary = vocabulary['article_sentence'].tolist()
vocabulary = np.array(vocabulary)


class SentimentAnalysis:
    def __init__(self):
        self.max_len = 50
        self.model = Model()

        self.tokenizer = Tokenizer(num_words=60000)
        self.tokenizer.fit_on_texts(vocabulary)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def tokenize(self, feature):
        feature = self.tokenizer.texts_to_sequences(feature)
        feature = pad_sequences(feature, padding='post', maxlen=self.max_len)

        return feature

    def get_embedding(self):
        word2vec = get_word2vec_model()
        embedding_matrix = zeros((self.vocab_size, 300))

        for word, index in self.tokenizer.word_index.items():
            try:
                embedding_vector = word2vec.get_vector(word)
                embedding_matrix[index] = embedding_vector
            except KeyError:
                print(word)

        return embedding_matrix

    def initialize_model(self):

        embedding_matrix = self.get_embedding()

        input_layer = Input(shape=(self.max_len,), dtype=tf.int32)

        embedding_layer = Embedding(self.vocab_size, 300, weights=[embedding_matrix], input_length=self.max_len,
                                    trainable=False)(
            input_layer)

        lstm_one = Bidirectional(LSTM(units=128, return_sequences=True))(embedding_layer)
        dropout_layer_one = Dropout(0.3)(lstm_one)

        lstm_two = Bidirectional(LSTM(units=128, return_sequences=False))(dropout_layer_one)
        dropout_layer_two = Dropout(0.3)(lstm_two)

        dense_layer_one = Dense(128, activation='relu')(dropout_layer_two)
        dropout_layer_four = Dropout(0.3)(dense_layer_one)

        dense_layer_two = Dense(128, activation='relu')(dropout_layer_four)
        dropout_layer_five = Dropout(0.3)(dense_layer_two)

        dense_layer_three = Dense(128, activation='relu')(dropout_layer_five)
        dropout_layer_six = Dropout(0.3)(dense_layer_three)

        dense_layer_four = Dense(128, activation='relu')(dropout_layer_six)
        dropout_layer_seven = Dropout(0.3)(dense_layer_four)

        dense_layer_five = Dense(128, activation='relu')(dropout_layer_seven)
        dropout_layer_eight = Dropout(0.3)(dense_layer_five)

        output_layer = Dense(1, activation='sigmoid')(dropout_layer_eight)

        model = Model(inputs=[input_layer], outputs=output_layer)

        sgd = SGD(learning_rate=0.1, momentum=0.2, decay=0.001)

        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        self.model = model

    def evaluate_model(self, feature, label):

        x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.20, random_state=42)

        # history = self.model.fit(x_train, y_train, callbacks=[LearningRateReducerCb()], batch_size=2, epochs=20,verbose=1, validation_split=0.2)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        history = self.model.fit(x_train, y_train, batch_size=2, epochs=20, verbose=1, validation_split=0.2,
                                 callbacks=[es])

        score = self.model.evaluate(x_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train_model(self, feature, label):

        feature = self.tokenize(feature)

        mc = ModelCheckpoint(Config.sentiment_analysis_model, monitor='val_accuracy', mode='max', verbose=1,
                             save_best_only=True)

        history = self.model.fit(feature, label, callbacks=[mc], epochs=100, verbose=1, validation_split=0.2)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict(self, feature):
        label = self.model.predict(feature, verbose=1)
        return label


def load_model(self):
    self.model.load_weights(Config.sentiment_analysis_model)


stop_words = set(stopwords.words('nepali'))


def get_word2vec_model():
    return KeyedVectors.load_word2vec_format(Config.word2vec_embedding, binary=False)


def pre_process_text(sentence):
    # sentence = ' '.join([word for word in sentence.split() if word not in stop_words])

    sentence = sentence.split(' ')

    clean_sentences = list()

    for word in sentence:
        word = re.sub(r'[,/\–-—()?;:’\'"‘“”`]', ' ', word)
        word = re.sub(r'\s+', ' ', word)
        word = re.sub(r'\u200d', '', word)

        clean_sentences.append(word.strip())

    while '' in clean_sentences:
        clean_sentences.remove('')

    if len(clean_sentences) >= 10:
        # print(len(clean_sentences))
        clean_sentences = ' '.join(clean_sentences)
        return clean_sentences.strip()


def pre_process_sentiment(sentiment):
    if sentiment == 'N':
        return
    if sentiment == 'Ne':
        return 0
    if sentiment == 'P':
        return 1


if __name__ == '__main__':
    sentiment_labelled_data = pd.read_excel(Config.sentiment_labelled_data, sheetname='Sheet 1').fillna('')
    sentiment_labelled_data = sentiment_labelled_data[['article_sentence', 'sentiment']]

    sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['article_sentence'] != '']
    sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['sentiment'] != '']

    sentiment_labelled_data['article_sentence'] = sentiment_labelled_data['article_sentence'].apply(pre_process_text, 1)
    sentiment_labelled_data['sentiment'] = sentiment_labelled_data['sentiment'].apply(pre_process_sentiment, 1)

    sentiment_labelled_data = sentiment_labelled_data.fillna('')

    sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['article_sentence'] != '']
    sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['sentiment'] != '']

    sentiment_labelled_data = shuffle(sentiment_labelled_data)

    print(sentiment_labelled_data.shape)

    X_feature = sentiment_labelled_data['article_sentence'].tolist()

    Y_label = sentiment_labelled_data['sentiment'].tolist()

    X_feature = np.array(X_feature)
    Y_label = np.array(Y_label)

    sentiment_analysis = SentimentAnalysis()
    sentiment_analysis.initialize_model()
    sentiment_analysis.train_model(X_feature, Y_label)
