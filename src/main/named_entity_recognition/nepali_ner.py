from config.config import Config
import nltk
from nltk.tag.stanford import StanfordNERTagger


class NepaliNER:
    def __init__(self):
        print('\nInitializing NER')
        self.ner_tagger = StanfordNERTagger(Config.ner_model, Config.ner_jar, encoding='utf8')

    def tag(self, text):
        print('\nTagging NER')

        tagged_words = self.ner_tagger.tag(nltk.word_tokenize(text))
        ner_tags = [data[1] for data in tagged_words]

        return ner_tags
