import pandas as pd
from nltk.corpus import indian
from sklearn.utils import shuffle

from config.config import Config
from src.main.pos_tagger.nepali_pos_tagger import NepaliPoSTagger
from src.main.pre_processing.article_cleaner import ArticleCleaner
from src.main.pre_processing.stemmer import Stemmer

from src.main.named_entity_recognition.nepali_ner import NepaliNER

from src.main.anaphora_resolution.anaphora_resolution import AnaphoraResolution


def pre_processing_clean_article(article):
    print('\nCleaning Article')

    # Remove Punctuation

    article_cleaner = ArticleCleaner(article)
    article_cleaner.remove_special_characters()
    article = article_cleaner.get_clean_article()

    print(article)

    return article


def pre_processing_stemmer(article):
    print('\nStemming Article')
    stemmer = Stemmer()
    article = ' '.join(stemmer.get_suffix(article.strip()))
    print(article)
    return article


def pipeline(article):
    # Pre-Processing

    article = pre_processing_clean_article(article)
    article = pre_processing_stemmer(article)

    # Part-of-Speech Tagging
    # Tagging
    article, pos_tags = nepali_pos_tagger.tag(article)

    # nepali_pos_tagger.translate_unk(article).to_csv('pos_tagged_politician_data.csv', index=False,encoding='utf-8')

    # Named Entity Recognition
    # Tagging
    ner_tags = nepali_ner.tag(' '.join(article))

    combined_article = ' '

    # print('\n{:15}{:5} {}'.format('word', 'pos', 'ner'))
    for word, pos, ner in zip(article, pos_tags, ner_tags):
        # print('{:15} {:5} {}'.format(word, pos, ner))
        combined_article = combined_article.strip() + ' ' + word.strip() + '_' + pos.strip() + '_' + ner.strip() + ' '

    ner_df = pd.DataFrame()
    ner_df['word'] = article
    ner_df['pos_tag'] = pos_tags
    ner_df['ner'] = ner_tags

    ner_df.to_csv('ner.tsv', index=False, encoding='utf-8', sep='\t')

    print(combined_article)

    # Anaphora Resolution
    anaphora_resolution = AnaphoraResolution()
    article = anaphora_resolution.resolve_anaphora(combined_article)

    print(article)

    return article


# noinspection PyBroadException
def process_articles(row):
    print('.......................................')
    print(row.name)

    print('\nArticle : {}'.format(row['article']))

    try:

        return pipeline(row['article'])

    except Exception:
        print(row['article_id'])


def training_politicians():
    # 37
    politician_article = pd.read_csv(Config.politician_training_file, dtype=object, encoding='utf-8').fillna('')

    politician_article = politician_article.drop_duplicates(subset=['article'], keep='last')
    politician_article['length'] = politician_article['article'].str.len()
    politician_article = politician_article.sort_values('length', ascending=True)
    politician_article = politician_article[politician_article['article'] != '']

    return politician_article[0]


# माइतीघर	 मण्डला

if __name__ == '__main__':
    sentiment_training_data = pd.read_csv(Config.sentiment_training_data, dtype=object, encoding='utf-8').fillna('')

    # Part-of-Speech Initialization
    nepali_pos_tagger = NepaliPoSTagger()

    # nepali_corpora = indian.tagged_sents('nepali.pos')
    # nepali_pos_tagger.train_model(nepali_corpora)
    # nepali_pos_tagger.save_model()

    # nepali_corpora = indian.tagged_sents('nepali.pos')
    # nepali_corpora = shuffle(nepali_corpora)
    # nepali_pos_tagger.five_fold_validation(nepali_corpora)

    nepali_pos_tagger.load_model()

    # Named Entity Recognition Initialization
    nepali_ner = NepaliNER()

    # pipeline('नेपालको प्रधानमन्त्रीको नाम केपी शर्मा ओली हो । उनी एक असल नेता हुन ।')

    # exit()

    sentiment_training_data['length'] = sentiment_training_data['article'].str.len()
    sentiment_training_data = sentiment_training_data.sort_values('length', ascending=True)

    print(sentiment_training_data.shape)

    sentiment_training_data['category'] = sentiment_training_data['category'].apply(lambda x: x.lower(), 1)

    print(list(set(list(sentiment_training_data['article_source']))))

    sentiment_training_data = sentiment_training_data[sentiment_training_data['category'] == 'opinion']

    print(list(set(list(sentiment_training_data['article_source']))))

    print(sentiment_training_data.shape)

    sentiment_training_data = sentiment_training_data[sentiment_training_data['length'] >= 200]
    sentiment_training_data = sentiment_training_data[sentiment_training_data['length'] <= 5000]

    print(sentiment_training_data.shape)

    # sentiment_training_data = sentiment_training_data[:50]

    # sentiment_training_data = sentiment_training_data[sentiment_training_data['article_id'] == '13319']

    print(sentiment_training_data.shape)

    sentiment_training_data['resolved_article'] = sentiment_training_data.apply(process_articles, 1)
    sentiment_training_data.to_csv('anaphora_resolved_1.csv', index=False, encoding='utf-8')
