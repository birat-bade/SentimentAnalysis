import nltk
import pandas as pd
import pickle
import numpy as np

from googletrans import Translator
from nltk.tag import tnt
from sklearn import metrics
from config.config import Config


class NepaliPoSTagger:
    def __init__(self):
        print('\nInitializing PoS Tagger')
        self.tnt_pos_tagger = tnt.TnT()

    def reinitialize_model(self):
        self.tnt_pos_tagger = tnt.TnT()

    def train_model(self, corpora):
        print('Training Model')
        self.tnt_pos_tagger.train(corpora)

    def tag(self, text):
        print('\nTagging PoS')
        tagged_words = (self.tnt_pos_tagger.tag(nltk.word_tokenize(text)))

        combined_tagged_words_list = list()

        tagged_tag_list = list()
        tagged_word_list = list()

        for data in tagged_words:
            combined_tagged_words_list.append('{}_{}'.format(data[0], data[1]))

            tagged_word_list.append(data[0])
            tagged_tag_list.append(data[1])

        print(' '.join(combined_tagged_words_list))

        return tagged_word_list, tagged_tag_list

    def load_model(self):
        pickle_file = open(Config.pos_tagger_model, 'rb')
        self.tnt_pos_tagger = pickle.load(pickle_file)

    def save_model(self):
        print('Saving Model')
        pickle_file = open(Config.pos_tagger_model, 'wb')
        pickle.dump(self.tnt_pos_tagger, pickle_file)
        pickle_file.close()

    def five_fold_validation(self, corpora):
        print('K-fold Validation')
        ten_fold = np.array_split(corpora, 5)

        for i in range(0, 5):
            print('\n{}-fold'.format(str(i + 1)))
            self.reinitialize_model()
            fold_index_list = [j for j in range(0, 5)]
            fold_index_list.remove(i)
            nine_fold = list()

            for index in fold_index_list:
                nine_fold += list(ten_fold[index])

            one_fold = list(ten_fold[i])

            self.train_model(nine_fold)
            accuracy = self.tnt_pos_tagger.evaluate(one_fold)
            print('Accuracy :{}'.format(accuracy))
            self.validate_metrics(one_fold)

    def validate_metrics(self, test_corpora):
        print('Calculating Metrics')

        tagged_test_sentences = self.tnt_pos_tagger.tag_sents([[token for token, tag in sent] for sent in test_corpora])
        gold = [str(tag) for sentence in test_corpora for token, tag in sentence]
        prediction = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]

        print(metrics.classification_report(gold, prediction))

    def translate_unk(self, string):
        tagged_news_data_set = pd.DataFrame(columns=['tokens', 'previous_tags', 'current_tags', 'translation'])

        tagged_words = (self.tnt_pos_tagger.tag(nltk.word_tokenize(string)))
        token_list, previous_tag_list, current_tag_list, translation_list = translate_unknown(tagged_words)

        tagged_news_data_set.tokens = token_list
        tagged_news_data_set.previous_tags = previous_tag_list
        tagged_news_data_set.current_tags = current_tag_list
        tagged_news_data_set.translation = translation_list

        return tagged_news_data_set


def translate_unknown(tagged_words):
    translator = Translator()

    token_list = list()
    previous_tag_list = list()
    current_tag_list = list()
    translation_list = list()

    for token, tag in tagged_words:

        previous_tag_list.append(tag)

        translation = 'N/R'

        if tag == 'Unk':
            translated = translator.translate(token)
            translated_token = str(translated.text)

            translated_pos = nltk.tag.pos_tag([translated_token])

            tag = '<' + translated_pos[0][1] + '>'
            translation = translated_pos[0][0]

        token_list.append(token)
        current_tag_list.append(tag)
        translation_list.append(translation)

    return token_list, previous_tag_list, current_tag_list, translation_list


corpus = pd.read_csv(Config.corpus, sep='|')
corpus = corpus.drop_duplicates(['data'], keep='last')

data_set = pd.DataFrame()
corpora_list = list()
sentence_counter = 0


def prepare_corpora_from_tagged_corpus():
    print('Creating Corpora')
    corpora_list.append(Config.corpora_head)
    corpus.apply(_process_corpus, 1)
    corpora_list.append(Config.corpora_tail)


def _process_corpus(row):
    global sentence_counter
    global data_set
    df_temp = pd.DataFrame(columns=['tokens', 'tags'])

    data = row['data']
    temp = data.split(' ')

    word_list = list()
    tag_list = list()

    for word in temp:
        if '<' in word:
            if word.count('<') > 1:
                pos_temp = word.split('<')

                first_pos_word = pos_temp[0]
                pos_tag = pos_temp[1].split('>')[0]

                word_list.append(first_pos_word)
                tag_list.append(pos_tag)

                suffix_tag_list = list()

                for i in range(1, (len(pos_temp))):
                    temp_suffix = pos_temp[i].split('>')
                    for suffix in temp_suffix:
                        suffix_tag_list.append(suffix)

                suffix_tag_list = suffix_tag_list[1:len(suffix_tag_list) - 1]

                for i in range(0, len(suffix_tag_list)):
                    if (i + 1) % 2 == 1:
                        word_list.append(suffix_tag_list[i])
                    else:
                        tag_list.append(suffix_tag_list[i])
            else:

                pos_temp = word.split('<')
                pos_word = pos_temp[0]
                pos_tag = pos_temp[1].replace('>', '')

                word_list.append(pos_word)
                tag_list.append(pos_tag)

    sentence_counter += 1
    sentence = list()

    for i in range(0, len(word_list)):
        sentence.append(word_list[i] + '_' + tag_list[i])
    sentence = '<Sentence id=' + str(sentence_counter) + '>\n' + ' '.join(sentence) + '\n</Sentence>'
    corpora_list.append(sentence)

    df_temp['tokens'] = word_list
    df_temp['tags'] = tag_list
    data_set = data_set.append(df_temp)


def save_training_data_set():
    corpus.to_csv(Config.training_data_set, index=False, encoding='utf-8')


def save_corpora():
    print('Saving Corpora')
    corpora = '\n'.join(corpora_list)
    nepali_corpora = open(Config.nepali_corpora, "wb")
    nepali_corpora.write(corpora.encode('utf-8'))
    nepali_corpora.close()


if __name__ == '__main__':
    print('\nInitializing Corpus')
    prepare_corpora_from_tagged_corpus()
    save_training_data_set()
    save_corpora()
