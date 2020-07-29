class Config:
    corpus = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\pos_tagger\\combine.csv'
    training_data_set = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\pos_tagger\\training_data_set.csv'
    pos_tagger_model = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\pos_tagger\\pos_tagger_model.pickle'

    tagged_news_data_set = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\tagged_news_data_set.csv'

    log_path = 'log\\logs.log'

    article = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\pos_tagger\\article.csv'

    nepali_suffix = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\stemmer\\raw_nepali_suffix.csv'
    processed_nepali_suffix = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\stemmer\\processed_nepali_suffix.csv'
    skip_words = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\stemmer\\skip_words.csv'

    politician_training_file = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\pos_tagger\\politician_training_article.csv'

    server = 'localhost'
    db = 'article_warehouse'
    user = 'root'
    password = ''

    nepali_corpora = 'C:\\Users\\DELL\\AppData\\Roaming\\nltk_data\\corpora\\indian\\nepali.pos'

    corpora_head = '<Corpora type="Monolingual-POS-TAGGED" Language="Nepali">'
    corpora_tail = '<\\Corpora>'

    ner_training_file = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\ner\\total.bio'

    ner_model = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\ner\\dummy-ner-model-nepali.ser.gz'
    ner_jar = 'F:\\Projects\\Python Projects\\SentimentAnalysis\\files\\ner\\stanford-ner-2018-10-16\\stanford-ner.jar'

    sentiment_raw_training_data = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\sentiment_analysis_plotting_dataset_raw.csv'
    sentiment_labelled_data = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\sentiment_labelled_data.xlsx'

    sentiment_testing_data_anaphora_resolved = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\anaphora_resolved.csv'
    sentiment_testing_data = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\training_data.csv'

    sentiment_analysis_model = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\sentiment_analysis_model.h5'

    labels = ['2018/04', '2018/05', '2018/06', '2018/07', '2018/08', '2018/09', '2018/10', '2018/11', '2018/12',
              '2019/01', '2019/02', '2019/03', '2019/04']

    labels_plot = ['2018/04', '2018/05', '2018/06', '2018/07', '2018/08', '2018/09', '2018/10', '2018/11', '2018/12',
                   '2019/01', '2019/02', '2019/03', '2019/04', '']

    x1_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    plotting_dataset = 'F:\\Projects\Python Projects\\SentimentAnalysis\\files\\sentiment_analysis\\plotting_data.csv'
