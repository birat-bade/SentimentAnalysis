import os
import re

import pandas as pd

from gensim.models import Word2Vec

from config.config import Config

all_files = os.listdir('F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis')
all_sentences = list()


def pre_process_text(sentence):
    sentence = sentence.split(' ')

    clean_sentences = list()

    for word in sentence:
        word = re.sub(r'[,/–-—()?;:’\'"‘“”`.]', ' ', word)
        word = re.sub(r'\s+', ' ', word)
        word = re.sub(r'\u200d', '', word)

        clean_sentences.append(word.strip())

    temp = ' '.join(clean_sentences)
    clean_sentences = temp.split(' ')

    while '' in clean_sentences:
        clean_sentences.remove('')
    while ' ' in clean_sentences:
        clean_sentences.remove(' ')

    all_sentences.append(clean_sentences)


def process_plotting_dataset():
    training_df = pd.DataFrame()

    for file in all_files:
        if 'training_data' in file:
            temp_df = pd.read_csv(
                'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\' + file,
                encoding='utf-8', dtype=object).fillna('')

            training_df = training_df.append(temp_df)

    training_df = training_df.fillna('')
    training_df = training_df[training_df['article_sentence'] != '']
    training_df = training_df.drop_duplicates(keep='first', subset='article_sentence')
    training_df['article_sentence'].apply(pre_process_text, 1)


def process_training_dataset():
    sheets = ['Sheet 1']

    for sheet in sheets:
        sentiment_labelled_data = pd.read_excel(Config.sentiment_labelled_data, sheetname=sheet).fillna('')
        sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['article_sentence'] != '']
        sentiment_labelled_data = sentiment_labelled_data[sentiment_labelled_data['sentiment'] != '']
        sentiment_labelled_data['article_sentence'].apply(pre_process_text, 1)


if __name__ == '__main__':
    process_plotting_dataset()
    process_training_dataset()

    model = Word2Vec(all_sentences, min_count=1, size=300, sg=1, negative=15, window=10)
    model.wv.save_word2vec_format(Config.word2vec_embedding, binary=False)

    all_sentences = [' '.join(sent) for sent in all_sentences]

    vocabulary = pd.DataFrame()
    vocabulary['article_sentence'] = all_sentences
    vocabulary.to_csv(Config.vocabulary, encoding='utf-8', index=False)
