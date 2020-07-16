import operator

from src.main.anaphora_resolution.discourse_class import DiscourseClass

sen = 'काठमाडौँ_NNP_LOC —_SYM_O नयाँ_JJ_ORG शक्ति_NN_ORG पार्टी_NN_ORG का_PKO_O संयोजक_NN_O बाबुराम_NNP_PER भट्टराई_NNP_PER ले_PLE_O नेकपा_NNP_ORG अध्यक्ष_NN_O पुष्पकमल_NNP_PER दाहाल_NNP_PER र_CC_O सीके_NNP_PER राउत_NNP_PER लाई_PLAI_O उस्तै_JJ_O हो_VBX_O भन्न_VBI_O नमिल्ने_VBNE_O बताएका_VBKO_O छन्_VBX_O ।_YF_O पार्टी_NN_ORG कार्यालय_NN_ORG मा_POP_O सोमबार_NNP_O आयोजित_NN_O पत्रकार_NN_O सम्मेलन_NN_O मा_POP_O बोल्दै_VBO_O संयोजक_NN_O बाबुराम_NNP_PER भट्टराई_NNP_PER ले_PLE_O दाहाल_NNP_PER को_PKO_O बचाउ_NN_O गर्दै_VBO_O उनी_PP_O एउटा_CL_O जनयुद्ध_NN_O को_PKO_O नेतृत्व_NN_O गरेर_VBO_O आएका_VBKO_O ब्यक्ति_NN_O रहेको_VBKO_O बताए_VBF_O ।_YF_O'

# personal_pronouns = ['उनी', 'उन', 'तिनि', 'यिनी', 'उनै', 'तपार्इँ', 'यस', 'म',
#                      'यिन', 'उहाँ', 'उहा', 'उ', 'तिन', 'ऊ', 'वहाँ', 'तपाई', 'हामी', 'तिमी', 'तिनी', 'मै']

personal_pronouns = ['उनी', 'उन', 'उनै', 'उहाँ', 'उहा', 'उ', 'ऊ', 'वहाँ']


# noinspection PyMethodMayBeStatic
class AnaphoraResolution:
    def __init__(self):
        self.counter = list()

    def resolve_anaphora(self, article):

        temp_article = list()

        flag = False

        discourse_object_list = self.process_discourse(article)

        discourse_model = dict()

        closest_per = ''

        for discourse_object in discourse_object_list:

            if discourse_object.get_token_ner() == 'PER':

                # closest_per = discourse_object.get_token()

                score = discourse_model.get(discourse_object.get_token())

                if score is None:
                    score = 0

                if discourse_object.get_next_token_pos() == 'PLE':
                    discourse_model.update({discourse_object.get_token(): score + 80})

                if discourse_object.get_next_token_pos() == 'PLAI':
                    discourse_model.update({discourse_object.get_token(): score + 70})
                    closest_per = discourse_object.get_token()

                if discourse_object.get_next_token_pos() == 'CC':
                    discourse_model.update({discourse_object.get_token(): score + 70})

                if discourse_object.get_next_token_pos() == 'PKO':
                    discourse_model.update({discourse_object.get_token(): score + 70})
                    closest_per = discourse_object.get_token()

                if discourse_object.get_next_token_pos() == 'VBX':
                    discourse_model.update({discourse_object.get_token(): score + 70})
                    closest_per = discourse_object.get_token()

            if discourse_object.get_token_pos() == 'YF':
                for key, value in discourse_model.items():
                    discourse_model.update({key: value / 2})

            if discourse_object.get_token() in personal_pronouns:

                if discourse_object.get_next_token_pos() != 'HRU':

                    score = discourse_model.get(closest_per)

                    if score is None:
                        score = 0

                    discourse_model.update({closest_per: score + 100})

                    possible_antecedent = max(discourse_model.items(), key=operator.itemgetter(1))[0]

                    discourse_object.set_token(possible_antecedent)

        return ' '.join([discourse_object.get_token() for discourse_object in discourse_object_list])

    def process_discourse(self, article):
        temp = article.strip().split(' ')

        print(len(temp))

        while '' in temp:
            temp.remove('')

        while '' in temp:
            temp.remove('')

        prev_per = str
        discourse_object_list = list()
        first_name_surname_list = list()

        for i in range(0, len(temp)):

            token = temp[i].split('_')[0]
            pos = temp[i].split('_')[1]
            ner = temp[i].split('_')[2]

            if temp[i].split('_')[2] == 'PER':
                if prev_per == 'PER':
                    token = discourse_object_list[-1].get_token() + ' ' + temp[i].split('_')[0]
                    discourse_object_list = discourse_object_list[:-1]
                    self.counter = self.counter[:-1]

                    first_name_surname_list.append(token.split(' '))

                for k in first_name_surname_list:
                    if token == k[-1]:
                        token = ' '.join(k)
                        break
                self.counter.append(token)

            try:
                next_token = temp[i + 1].split('_')[0]
                next_token_pos = temp[i + 1].split('_')[1]
                next_token_ner = temp[i + 1].split('_')[2]

                discourse_object = DiscourseClass(i, token, pos, ner, next_token, next_token_pos, next_token_ner)

            except IndexError:

                discourse_object = DiscourseClass(i, token, pos, ner, 'E.O.S', 'E.O.S', 'E.O.S')

            discourse_object_list.append(discourse_object)

            prev_per = temp[i].split('_')[2]

        print('\n{}'.format(list(set(self.counter))))

        if len(list(set(self.counter))) <= 3:
            return discourse_object_list

        return []

# print('{}  {}  {}  {}  {}  {}  {}'.format(discourse_object.get_index(), discourse_object.get_token(),
#                                           discourse_object.get_token_pos(), discourse_object.get_token_ner(),
#                                           discourse_object.get_next_token(),
#                                           discourse_object.get_next_token_pos(),
#                                           discourse_object.get_next_token_pos()))
