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

        discourse_sentences = self.process_discourse(article)

        discourse_model = dict()
        closest_per_dict = dict()
        adjust_factor_dict = dict()

        for sentence in discourse_sentences:

            closest_per = ''
            closest_per_next_token_pos = ''
            number_of_entities = get_number_of_entities(sentence)

            for token_object in sentence:

                # print(token_object.get_token())

                if token_object.get_token_ner() == 'PER':

                    closest_per = token_object.get_token()
                    closest_per_next_token_pos = token_object.get_next_token_pos()

                    if closest_per_next_token_pos == 'PLE' or closest_per_next_token_pos == 'PLAI':
                        closest_per_dict.update({closest_per: closest_per_next_token_pos})

                    if not (closest_per_dict.get(closest_per) == 'PLE' or closest_per_dict.get(closest_per) == 'PLAI'):
                        closest_per_dict.update({closest_per: closest_per_next_token_pos})

                    adjust_factor_dict.update({closest_per: 2})
                    score = discourse_model.get(token_object.get_token())

                    if score is None:
                        score = 0

                    salience_factor = calculate_salience_factor(token_object)
                    discourse_model.update({token_object.get_token(): score + salience_factor})

                if token_object.get_token() in personal_pronouns:
                    number_of_entities_in_front = get_number_of_entities(sentence[:sentence.index(token_object)])
                    if token_object.get_next_token_pos() != 'HRU':
                        if len(discourse_model) != 0:

                            # print(closest_per_dict)
                            # print(discourse_model)
                            # print(adjust_factor_dict)

                            possible_antecedent = max(discourse_model.items(), key=operator.itemgetter(1))[0]
                            if number_of_entities_in_front == 1:
                                temp_score = discourse_model.pop(closest_per, None)
                                try:
                                    possible_antecedent = max(discourse_model.items(),
                                                              key=operator.itemgetter(1))[0]
                                except ValueError:
                                    possible_antecedent = closest_per
                                discourse_model.update({closest_per: temp_score})
                            else:
                                if not in_same_sentence(sentence, token_object.get_token(), possible_antecedent):
                                    if next_pos_is_le_lai(token_object.get_next_token_pos(),
                                                          closest_per_dict.get(possible_antecedent)):
                                        temp_key = possible_antecedent
                                        antecedent_contenders = get_contender_antecedents(closest_per_dict,
                                                                                          token_object.get_next_token_pos())
                                        temp_score = list()
                                        for antecedent in antecedent_contenders:
                                            temp_score.append(discourse_model.get(antecedent))

                                        try:
                                            max_score = max(list(set(temp_score)))
                                            possible_antecedent = antecedent_contenders[temp_score.index(max_score)]
                                        except ValueError:
                                            possible_antecedent = temp_key

                                            # discourse_model.update({temp_key: temp_score})

                            token_object.set_token(possible_antecedent)
                            token_object.set_token_ner('PER')

                            score = discourse_model.get(token_object.get_token())
                            salience_factor = calculate_salience_factor(token_object)
                            discourse_model.update({token_object.get_token(): score + salience_factor})
                            closest_per = possible_antecedent

            for key, value in discourse_model.items():
                discourse_model.update({key: value / adjust_factor_dict.get(key)})

            if len(adjust_factor_dict) > 0:

                for key, value in adjust_factor_dict.items():
                    value += 0.7
                    adjust_factor_dict.update({key: value})

        named_entities = list(set(list(discourse_model.keys())))

        anaphora_resolved_sentences = list()

        for sentence_list in discourse_sentences:
            all_words = [data.get_token() for data in sentence_list]
            sentence = ' '.join(all_words)

            for named_entity in named_entities:
                if named_entity in sentence:
                    anaphora_resolved_sentences.append(sentence)
                    break

        return ' । '.join(anaphora_resolved_sentences)

    def process_discourse(self, article):

        all_sentences = article.strip().split('YF_O')

        while '' in all_sentences:
            all_sentences.remove('')

        while ' ' in all_sentences:
            all_sentences.remove(' ')

        discourse_sentences = list()

        first_name_surname_list = list()

        for sentence in all_sentences:

            temp = sentence.strip().split(' ')
            # print(len(temp))

            while '' in temp:
                temp.remove('')

            while ' ' in temp:
                temp.remove(' ')

            prev_per = str
            discourse_object_list = list()

            for i in range(0, len(temp)):

                if len(temp[i].split('_')) < 3:
                    break

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
            discourse_sentences.append(discourse_object_list)
            # print('\n{}'.format(list(set(self.counter))))

        if len(list(set(self.counter))) <= 3:
            return discourse_sentences

        return []


def get_contender_antecedents(closest_per_dict, next_token_pos):
    antecedents_contenders = list()
    for key, value in closest_per_dict.items():
        if value == next_token_pos:
            antecedents_contenders.append(key)
    return antecedents_contenders


def next_pos_is_le_lai(token_pos, possible_antecedent_pos):
    pos_list = ['PLE', 'PLAI']

    if token_pos in pos_list:
        if token_pos != possible_antecedent_pos:
            return True
    return False


def in_same_sentence(sentence, token, antecedent):
    all_words = [data.get_token() for data in sentence]
    all_words = all_words[:all_words.index(token)]
    if antecedent in all_words:
        return True


def get_number_of_entities(sentence):
    count = 0
    for token_object in sentence:
        if token_object.get_token_ner() == 'PER':
            count += 1
    return count


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
