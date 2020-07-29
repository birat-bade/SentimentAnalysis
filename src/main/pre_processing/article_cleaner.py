class ArticleCleaner:
    def __init__(self, article):
        self.article = article
        self.article = self.article.split('।')

    def remove_special_characters(self):
        self.article = [
            string.strip().replace('\\xa0', '').replace('\\u202f', '').replace('\\u200d', '').replace('\r\n',
                                                                                                      '').replace(
                '\\r\\n', ' ').replace('\'u200c', '').replace('\'', '\' ') for
            string in self.article]

    def get_clean_article(self):
        self.article = self.article[:5]
        self.article = ' । '.join(self.article)
        if 'उनी' not in self.article:
            return
        return self.article
