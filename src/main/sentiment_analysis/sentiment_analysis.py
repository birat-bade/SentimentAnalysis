import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import tensorflow_hub as hub

import tensorflow as tf

from config.config import Config

from sklearn.model_selection import train_test_split

from src.main.sentiment_analysis.callback import LearningRateReducerCb

x = tf.compat.v1

x.disable_eager_execution()

keras = tf.keras

Sequential = tf.keras.Sequential

Embedding = tf.keras.layers.Embedding

LSTM = tf.keras.layers.LSTM

Bidirectional = tf.keras.layers.Bidirectional

Dense = keras.layers.Dense

Dropout = keras.layers.Dropout

Input = keras.layers.Input

Lambda = keras.layers.Lambda

Model = tf.keras.Model

SGD = tf.keras.optimizers.SGD

EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint


class SentimentAnalysis:
    def __init__(self):
        self.max_len = 50
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        self.model = Model()

    def elmo_embedding(self, sentence):
        return self.elmo(tf.squeeze(tf.cast(sentence, tf.string), axis=1), signature="default", as_dict=True)["elmo"]

    def initialize_model(self):
        input_layer = Input(shape=(1,), dtype=tf.string)

        embedding_layer = Lambda(self.elmo_embedding, output_shape=(1024,))(input_layer)

        bidirectional_lstm = Bidirectional(LSTM(units=256, return_sequences=True))(embedding_layer)

        dropout_layer_zero = Dropout(0.5)(bidirectional_lstm)

        lstm = LSTM(units=256, return_sequences=False)(dropout_layer_zero)

        dense_layer_one = Dense(8336, activation='relu')(lstm)

        dropout_layer_one = Dropout(0.5)(dense_layer_one)

        dense_layer_two = Dense(4168, activation='relu')(dropout_layer_one)

        dropout_layer_two = Dropout(0.5)(dense_layer_two)

        output_layer = Dense(1, activation='sigmoid')(dropout_layer_two)

        model = Model(input_layer, output_layer)

        sgd = SGD(learning_rate=0.1, momentum=0.2)

        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        self.model = model

    def evaluate_model(self, feature, label):
        with tf.compat.v1.Session() as session:
            tf.compat.v1.keras.backend.set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())

            x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.20,
                                                                random_state=42)

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
        with tf.compat.v1.Session() as session:
            tf.compat.v1.keras.backend.set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            mc = ModelCheckpoint(Config.sentiment_analysis_model, monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)

            # history = self.model.fit(feature, label, batch_size=2, epochs=20, verbose=1, validation_split=0.2,
            #                          callbacks=[LearningRateReducerCb(),es, mc])

            history = self.model.fit(feature, label, callbacks=[es, mc], batch_size=2,
                                     epochs=20, verbose=1, validation_split=0.2)

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])

            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def predict(self, feature):
        with tf.compat.v1.Session() as session:
            tf.compat.v1.keras.backend.set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())

            label = self.model.predict(feature, verbose=1)
            return label

    def load_model(self):
        self.model.load_weights(Config.sentiment_analysis_model)


stop_words = set(stopwords.words('nepali'))


def pre_process_text(sentence):
    # sentence = ' '.join([word for word in sentence.split() if word not in stop_words])

    sentence = re.sub(r'[,\-—()?;:’\'"]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = sentence.split(' ')

    if len(sentence) >= 10:
        sentence = ' '.join(sentence)

        if 'प्रकाशित' not in sentence:
            return sentence.strip()


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
