import pandas as pd
from nltk.corpus import indian
from sklearn.utils import shuffle

from config.config import Config
from src.main.pos_tagger.nepali_pos_tagger import NepaliPoSTagger
from src.main.pre_processing.article_cleaner import ArticleCleaner
from src.main.pre_processing.stemmer import Stemmer

from src.main.named_entity_recognition.nepali_ner import NepaliNER

from src.main.anaphora_resolution.anaphora_resolution import AnaphoraResolution


def pipeline(article):
    # Pre-Processing
    article_cleaner = ArticleCleaner(article)
    article_cleaner.remove_special_characters()

    stemmer = Stemmer()
    article = article_cleaner.get_clean_article()
    article = ' '.join(stemmer.get_suffix(article.strip()))

    # Part-of-Speech Tagging
    article, pos_tags = nepali_pos_tagger.tag(article)
    # nepali_pos_tagger.translate_unk(article).to_csv('pos_tagged_politician_data.csv', index=False,encoding='utf-8')

    # Named Entity Recognition
    ner_tags = nepali_ner.tag(' '.join(article))
    article_pos_ner = ' '

    # print('\n{:15}{:5} {}'.format('word', 'pos', 'ner'))
    for word, pos, ner in zip(article, pos_tags, ner_tags):
        # print('{:15} {:5} {}'.format(word, pos, ner))
        article_pos_ner = article_pos_ner.strip() + ' ' + word.strip() + '_' + pos.strip() + '_' + ner.strip() + ' '

    ner_df = pd.DataFrame()
    ner_df['word'] = article
    ner_df['pos_tag'] = pos_tags
    ner_df['ner'] = ner_tags

    # ner_df.to_csv('files/ner.tsv', index=False, encoding='utf-8', sep='\t')

    # Anaphora Resolution
    anaphora_resolution = AnaphoraResolution()
    article = anaphora_resolution.resolve_anaphora(article_pos_ner)

    print(article)

    return article


i = 0


# noinspection PyBroadException
def process_articles(row):
    global i
    print('.......................................')

    i += 1
    print(i)

    print(row['article_id'])
    try:
        return pipeline(row['article'])

    except Exception as e:
        print(str(e))


def training_politicians():
    # 37
    politician_article = pd.read_csv(Config.politician_training_file, dtype=object, encoding='utf-8').fillna('')

    politician_article = politician_article.drop_duplicates(subset=['article'], keep='last')
    politician_article['length'] = politician_article['article'].str.len()
    politician_article = politician_article.sort_values('length', ascending=True)
    politician_article = politician_article[politician_article['article'] != '']

    return politician_article[0]


def split_sentences(row):
    if row.resolved_article is not None:
        temp = row.resolved_article.split('।')

        temp_df = pd.DataFrame()

        temp_df['article_id'] = [str(row['article_id']) for data in temp]
        temp_df['article_url'] = [row['article_url'] for data in temp]
        temp_df['article_source'] = [row['article_source'] for data in temp]
        temp_df['category'] = [row['category'] for data in temp]
        temp_df['title'] = [row['title'] for data in temp]
        temp_df['date'] = [row['date'] for data in temp]
        temp_df['politician_name'] = [row['politician_name'] for data in temp]
        temp_df['article_sentence'] = [data.strip() for data in temp]

        global anaphora_sentence
        anaphora_sentence = anaphora_sentence.append(temp_df)


# माइतीघर	 मण्डला

if __name__ == '__main__':
    sentiment_training_data = pd.read_csv(Config.sentiment_raw_training_data, dtype=object, encoding='utf-8').fillna('')
    print(sentiment_training_data.shape)

    # Part-of-Speech Initialization
    nepali_pos_tagger = NepaliPoSTagger()
    # nepali_corpora = indian.tagged_sents('nepali.pos')
    # nepali_pos_tagger.train_model(nepali_corpora)
    # nepali_pos_tagger.save_model()
    nepali_pos_tagger.load_model()

    # Named Entity Recognition Initialization
    nepali_ner = NepaliNER()

    # pipeline('नेपालको प्रधानमन्त्रीको नाम केपी शर्मा ओली हो । उनी एक असल नेता हुन । राम मनोहर यादव निकिता पौडेलको विमलेन्द्र निधि विद्यासुन्दर शाक्य, मातृका यादव')
    # exit()

    sentiment_training_data['length'] = sentiment_training_data['article'].str.len()
    # sentiment_training_data = sentiment_training_data.sort_values('length', ascending=True)

    print(sentiment_training_data.shape)

    sentiment_training_data['category'] = sentiment_training_data['category'].apply(lambda x: x.lower(), 1)
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['category'] == 'opinion']
    sentiment_training_data = sentiment_training_data[sentiment_training_data['length'] <= 5000]

    print(sentiment_training_data.shape)

    # sentiment_training_data = sentiment_training_data[1200:]

    sentiment_training_data = sentiment_training_data.drop_duplicates(keep='first', subset='article_id')

    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '34538']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '13319']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '110999']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '10214']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '47158']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '25954']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '120503']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '41864']
    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '47121'] 13199 30397  109581 39402 78758 77904 16024 16946 14615 12209
    sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '119004']

    sentiment_training_data['resolved_article'] = sentiment_training_data.apply(process_articles, 1)
    sentiment_training_data = sentiment_training_data[sentiment_training_data['resolved_article'] != '']

    print(sentiment_training_data.shape)

    # sentiment_training_data.to_csv(Config.sentiment_testing_data_anaphora_resolved, index=False, encoding='utf-8')

    anaphora_sentence = pd.DataFrame()
    sentiment_training_data.apply(split_sentences, 1)

    anaphora_sentence = anaphora_sentence.fillna('')
    anaphora_sentence = anaphora_sentence[anaphora_sentence['article_sentence'] != '']

    anaphora_sentence.to_csv(Config.sentiment_testing_data, index=False, encoding='utf-8')
