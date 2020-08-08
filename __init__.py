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
    article = article_cleaner.get_clean_article()

    stemmer = Stemmer()
    article = ' '.join(stemmer.stem_article(article.strip()))

    # Part-of-Speech Tagging
    article, pos_tags = nepali_pos_tagger.tag(article)

    # Named Entity Recognition
    ner_tags = nepali_ner.tag(' '.join(article))
    article_pos_ner = ' '

    for word, pos, ner in zip(article, pos_tags, ner_tags):
        article_pos_ner = article_pos_ner.strip() + ' ' + word.strip() + '_' + pos.strip() + '_' + ner.strip() + ' '

    ner_df = pd.DataFrame()
    ner_df['word'] = article
    ner_df['pos_tag'] = pos_tags
    ner_df['ner'] = ner_tags

    # Anaphora Resolution
    anaphora_resolution = AnaphoraResolution()
    article = anaphora_resolution.resolve_anaphora(article_pos_ner)

    print(article)

    return article


def process_articles(row):
    try:
        return pipeline(row['article'])

    except Exception as e:
        print(str(e))


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


if __name__ == '__main__':
    sentiment_training_data = pd.read_csv(Config.sentiment_raw_training_data, dtype=object, encoding='utf-8').fillna('')

    # Part-of-Speech Initialization
    nepali_pos_tagger = NepaliPoSTagger()
    nepali_pos_tagger.load_model()

    # Named Entity Recognition Initialization
    nepali_ner = NepaliNER()

    sentiment_training_data['length'] = sentiment_training_data['article'].str.len()
    sentiment_training_data['category'] = sentiment_training_data['category'].apply(lambda x: x.lower(), 1)
    sentiment_training_data = sentiment_training_data[sentiment_training_data['length'] <= 5000]
    sentiment_training_data = sentiment_training_data.drop_duplicates(keep='first', subset='article_id')
    sentiment_training_data['resolved_article'] = sentiment_training_data.apply(process_articles, 1)
    sentiment_training_data = sentiment_training_data[sentiment_training_data['resolved_article'] != '']

    anaphora_sentence = pd.DataFrame()
    sentiment_training_data.apply(split_sentences, 1)
    anaphora_sentence.to_csv(Config.sentiment_testing_data, index=False, encoding='utf-8')
