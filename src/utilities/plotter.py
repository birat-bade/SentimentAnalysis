import os

import pandas as pd

import numpy as np

import re

import json

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from config.config import Config

from src.main.sentiment_analysis.sentiment_analysis import SentimentAnalysis

all_files = os.listdir('F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis')

date_dict = dict()

article_count_dict = dict()

frame_list = list()
x_oli = list()
y_oli = list()

plt.style.use('seaborn')


# plt.style.use('ggplot')


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
        clean_sentences = ' '.join(clean_sentences)

        return clean_sentences.strip()


def count_articles(row):
    key = str(row['date'])

    count = article_count_dict.get(key)
    if count is None:
        count = list()

    count.append(row['article_id'])
    article_count_dict.update({key: count})


def prepare_plotting_data(row):
    key = str(row['date'])

    count_sentiment = date_dict.get(key)
    if count_sentiment is None:
        count = 0
        sentiment = 0
    else:
        count = date_dict.get(key)[0]
        sentiment = date_dict.get(key)[1]

    count += 1
    sentiment += float(row['sentiment'])

    date_dict.update({key: (count, sentiment)})


def prepare_plotting_dataset():
    training_df = pd.DataFrame()

    for file in all_files:
        if 'training_data' in file:
            temp_df = pd.read_csv(
                'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\' + file,
                encoding='utf-8', dtype=object).fillna('')

            training_df = training_df.append(temp_df)

    training_df = training_df[training_df['article_sentence'] != '']
    training_df = training_df.drop_duplicates(keep='first', subset='article_sentence')

    training_df['article_sentence'] = training_df['article_sentence'].apply(pre_process_text, 1)
    training_df = training_df.fillna('')
    training_df = training_df[training_df['article_sentence'] != '']
    print(training_df.shape)

    feature = training_df['article_sentence'].tolist()
    feature = np.array(feature)

    sentiment_analysis = SentimentAnalysis()
    sentiment_analysis.initialize_model()
    sentiment_analysis.load_model()
    label = sentiment_analysis.predict(feature)

    training_df['sentiment'] = label

    training_df.to_csv(Config.plotting_dataset, index=False, encoding='utf-8')


def animate(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    x_oli.append(frame[0])
    y_oli.append(frame[1])

    ax1.plot(x_oli, y_oli, marker='o', linestyle='-.')
    ax1.set_xticks(Config.x1_plot)
    ax1.set_yticks([49, 49.2, 49.4, 49.6, 49.8, 50])
    ax1.set_xticklabels(Config.labels_plot, minor=False)

    plt.xlabel('Month')
    plt.ylabel('Approval Rating')
    plt.title('Approval Rating of Nepali Politicians')

    plt.gca().legend(('Sher Bahadur Deuwa', '', ''))


if __name__ == '__main__':

    # prepare_plotting_dataset()

    plot_df = pd.read_csv(Config.plotting_dataset, encoding='utf-8', dtype=object).fillna('')
    plot_df = plot_df[plot_df['politician_name'] == 'KP Oli']
    plot_df = plot_df.sort_values('date', ascending=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = fig.add_subplot(1, 1, 1)
    ax3 = fig.add_subplot(1, 1, 1)
    fig.canvas.set_window_title('Approval Rating')

    plot_df.apply(prepare_plotting_data, 1)
    plot_df.apply(count_articles, 1)

    i = 0

    for k, v in article_count_dict.items():
        print('{} {}'.format(k, len(list((set(v))))))

    for k, v in date_dict.items():
        approval_rating = v[1] / v[0] * 100

        plotting_param = (i, approval_rating)
        frame_list.append(plotting_param)

        date_dict.update({k: approval_rating})

        print('{} {}'.format(k, v))

        i += 1

    ani = animation.FuncAnimation(fig, animate, frames=frame_list, interval=1000, repeat=False)
    plt.show()
