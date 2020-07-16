class DiscourseClass:
    def __init__(self, index, token, token_pos, token_ner, next_token, next_token_pos, next_token_ner):
        self.index = index
        self.token = token
        self.token_pos = token_pos
        self.token_ner = token_ner

        self.next_token = next_token
        self.next_token_pos = next_token_pos
        self.next_token_ner = next_token_ner

    def get_index(self):
        return self.index

    def get_token(self):
        return self.token

    def get_token_pos(self):
        return self.token_pos

    def get_token_ner(self):
        return self.token_ner

    def get_next_token(self):
        return self.next_token

    def get_next_token_pos(self):
        return self.next_token_pos

    def get_next_token_ner(self):
        return self.next_token_ner

    def set_token(self, token):
        self.token = token
