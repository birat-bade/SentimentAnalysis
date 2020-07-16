class Config:
    corpus = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\pos_tagger\\combine.csv'
    training_data_set = '../../../files/pos_tagger/training_data_set.csv'
    pos_tagger_model = 'files/pos_tagger/pos_tagger_model.pickle'

    tagged_news_data_set = 'files/tagged_news_data_set.csv'

    sentiment_training_data = 'files/sentiment_analysis/training_data.csv'

    log_path = 'log/logs.log'

    article = 'files/pos_tagger/article.csv'

    nepali_suffix = 'files/stemmer/raw_nepali_suffix.csv'
    processed_nepali_suffix = 'files/stemmer/processed_nepali_suffix.csv'
    skip_words = 'files/stemmer/skip_words.csv'

    politician_training_file = 'files/pos_tagger/politician_training_article.csv'

    server = 'localhost'
    db = 'article_warehouse'
    user = 'root'
    password = ''

    nepali_corpora = 'C:\\Users\\DELL\\AppData\\Roaming\\nltk_data\\corpora\\indian\\nepali.pos'

    corpora_head = '<Corpora type="Monolingual-POS-TAGGED" Language="Nepali">'
    corpora_tail = '</Corpora>'

    ner_training_file = '../../../files/ner/total.bio'

    ner_model = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\ner\\stanford-ner-2018-10-16\\dummy-ner-model-nepali.ser.gz'
    ner_jar = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\ner\\stanford-ner-2018-10-16\\stanford-ner.jar'
