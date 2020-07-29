import operator

from src.main.anaphora_resolution.discourse_class import DiscourseClass

sen = 'काठमाडौँ_NNP_LOC —_SYM_O नयाँ_JJ_ORG शक्ति_NN_ORG पार्टी_NN_ORG का_PKO_O संयोजक_NN_O बाबुराम_NNP_PER भट्टराई_NNP_PER ले_PLE_O नेकपा_NNP_ORG अध्यक्ष_NN_O पुष्पकमल_NNP_PER दाहाल_NNP_PER र_CC_O सीके_NNP_PER राउत_NNP_PER लाई_PLAI_O उस्तै_JJ_O हो_VBX_O भन्न_VBI_O नमिल्ने_VBNE_O बताएका_VBKO_O छन्_VBX_O ।_YF_O पार्टी_NN_ORG कार्यालय_NN_ORG मा_POP_O सोमबार_NNP_O आयोजित_NN_O पत्रकार_NN_O सम्मेलन_NN_O मा_POP_O बोल्दै_VBO_O संयोजक_NN_O बाबुराम_NNP_PER भट्टराई_NNP_PER ले_PLE_O दाहाल_NNP_PER को_PKO_O बचाउ_NN_O गर्दै_VBO_O उनी_PP_O एउटा_CL_O जनयुद्ध_NN_O को_PKO_O नेतृत्व_NN_O गरेर_VBO_O आएका_VBKO_O ब्यक्ति_NN_O रहेको_VBKO_O बताए_VBF_O ।_YF_O'

# personal_pronouns = ['उनी', 'उन', 'तिनि', 'यिनी', 'उनै', 'तपार्इँ', 'यस', 'म',
#                      'यिन', 'उहाँ', 'उहा', 'उ', 'तिन', 'ऊ', 'वहाँ', 'तपाई', 'हामी', 'तिमी', 'तिनी', 'मै']

personal_pronouns = ['उनी', 'उन', 'उनै', 'उहाँ', 'उहा', 'उ', 'ऊ', 'वहाँ']


class AnaphoraResolution:
    def __init__(self):
        self.counter = list()

    def resolve_anaphora(self, article):

        print('\nResolving Anaphora')

        temp_article = list()

        flag = False

        discourse_object_list = self.process_discourse(article)

        discourse_model = dict()

        closest_per = ''
        closest_per_next_token_pos = ''

        for discourse_object in discourse_object_list:

            if discourse_object.get_token_ner() == 'PER':

                closest_per = discourse_object.get_token()
                closest_per_next_token_pos = discourse_object.get_next_token_pos()

                score = discourse_model.get(discourse_object.get_token())
                if score is None:
                    score = 0

                salience_factor = calculate_salience_factor(discourse_object)
                discourse_model.update({discourse_object.get_token(): score + salience_factor})

            if discourse_object.get_token_pos() == 'YF':

                closest_per = ''
                closest_per_next_token_pos = ''

                for key, value in discourse_model.items():
                    discourse_model.update({key: value / 2})

            if discourse_object.get_token() in personal_pronouns:

                if discourse_object.get_next_token_pos() != 'HRU':
                    if len(discourse_model) != 0:

                        # print(discourse_model)

                        if closest_per_next_token_pos == 'PLE':
                            temp_discourse_model = discourse_model
                            temp_discourse_model.pop(closest_per, None)
                            possible_antecedent = max(temp_discourse_model.items(), key=operator.itemgetter(1))[0]
                        else:
                            possible_antecedent = max(discourse_model.items(), key=operator.itemgetter(1))[0]

                        discourse_object.set_token(possible_antecedent)
                        discourse_object.set_token_ner('PER')

                        score = discourse_model.get(discourse_object.get_token())
                        salience_factor = calculate_salience_factor(discourse_object)
                        discourse_model.update({discourse_object.get_token(): score + salience_factor})

        # return ' '.join([discourse_object.get_token() for discourse_object in discourse_object_list])

        named_entities = list(set(list(discourse_model.keys())))

        all_sentences = ' '.join([discourse_object.get_token() for discourse_object in discourse_object_list]) \
            .split('।')

        sentence_with_politician_name = list()

        for s in all_sentences:
            for named_entity in named_entities:
                if named_entity in s:
                    sentence_with_politician_name.append(s)
                    break

        return '{}{}'.format('।'.join(sentence_with_politician_name), '।')

    def process_discourse(self, article):

        # print(article)

        temp = article.strip().split(' ')

        # print(len(temp))

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

        # print('\n{}'.format(list(set(self.counter))))

        if len(list(set(self.counter))) <= 3:
            return discourse_object_list

        return []


def calculate_salience_factor(discourse_object):
    if discourse_object.get_next_token_pos() == 'PLE':
        return 70

    elif discourse_object.get_next_token_pos() == 'PLAI':
        return 80

    elif discourse_object.get_next_token_pos() == 'CC':
        return 80

    elif discourse_object.get_next_token_pos() == 'PKO':
        return 80

    elif discourse_object.get_next_token_pos().startswith('VB'):
        return 80

    elif discourse_object.get_next_token_pos().startswith('RP'):
        return 80

    else:
        return 70

# print('{}  {}  {}  {}  {}  {}  {}'.format(discourse_object.get_index(), discourse_object.get_token(),
#                                           discourse_object.get_token_pos(), discourse_object.get_token_ner(),
#                                           discourse_object.get_next_token(),
#                                           discourse_object.get_next_token_pos(),
#                                           discourse_object.get_next_token_pos()))
