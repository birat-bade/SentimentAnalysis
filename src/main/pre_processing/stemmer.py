import pandas as pd
from nltk import word_tokenize

from config.config import Config


class Stemmer:
    def __init__(self):
        self.nepali_suffix = pd.read_csv(Config.nepali_suffix).fillna('')
        self.skip_words = pd.read_csv(Config.skip_words).fillna('')

        self.nepali_suffix['nepali_suffix'] = self.nepali_suffix['nepali_suffix'].apply(lambda x: x.strip(), 1)
        self.nepali_suffix['length'] = self.nepali_suffix['nepali_suffix'].str.len()

        self.nepali_suffix = self.nepali_suffix.sort_values('length', ascending=True)

    def get_suffix(self, string):
        stemmed_list = list()
        suffix_list = list(self.nepali_suffix['nepali_suffix'])
        skip_words_list = list(self.skip_words['skip_words'])

        for data in (word_tokenize(string)):
            stemmed_list.append(strip_suffix(data, suffix_list, skip_words_list))
        return stemmed_list


def strip_suffix(string, suffix_list, skip_words_list):
    stem = string.strip()
    present_suffix = list()

    check_ends_with(stem, suffix_list, present_suffix, skip_words_list)

    present_suffix.reverse()
    return ' '.join(present_suffix).strip()


def check_ends_with(string, local_list, present_suffix, skip_words_list):
    for data in local_list:
        # Binary Search
        if string in skip_words_list:
            break
        if string.endswith(data):
            present_suffix.append(data.strip())
            return check_ends_with(string[:string.rfind(data)], local_list, present_suffix, skip_words_list)
    present_suffix.append(string.strip())
