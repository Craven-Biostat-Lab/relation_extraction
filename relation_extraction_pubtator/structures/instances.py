import itertools

def list_correction(checklist,entity_a_text,entity_b_text):
    for i in range(len(checklist)):
        word = checklist[i]
        if word in entity_a_text and word in entity_b_text:
            checklist[i] = 'ENTITY_A_B'
        elif word in entity_a_text:
            checklist[i] = 'ENTITY_A'
        elif word in entity_b_text:
                checklist[i] = 'ENTITY_B'

    return checklist
class Instance(object):

    def __init__(self,sentence,label):
        self.sentence = sentence
        self.label = label
        self.between_words = []
        self.build_words_between_features()
        self.dependency_elements = sentence.dep_parse
        self.dependency_path =  ''
        self.dependency_words = []
        self.features = []
        self.build_dep_path_and_words()

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label

    def get_label(self):
        return self.label

    def build_dep_path_and_words(self):
        previous_word = 'START_ENTITY'
        dep_path = []
        dep_words = set()
        for element in self.dependency_elements:
            split_element = element.split('|')
            word_1 = split_element[0]
            type = split_element[1]
            word_2 = split_element[2]
            dep_words.add(word_1)
            dep_words.add(word_2)
            reverse = False
            if word_2 == previous_word:
                reverse = True

            if reverse is False:
                dep_path.append(type)
                previous_word = word_2
            else:
                dep_path.append('-' + type)
                previous_word = word_1


        dep_words.remove('START_ENTITY')
        dep_words.remove('END_ENTITY')
        self.dependency_words=list(dep_words)
        self.dependency_path = ' '.join(dep_path)

    def build_words_between_features(self):
        entity_1 = self.sentence.entity_1_text
        entity_2 = self.sentence.entity_2_text

        entity_1_start = int(self.sentence.entity_1_loc.split(',')[0])
        entity_1_end = int(self.sentence.entity_1_loc.split(',')[1])
        entity_2_start = int(self.sentence.entity_2_loc.split(',')[0])
        entity_2_end = int(self.sentence.entity_2_loc.split(',')[1])

        distance_1 = entity_1_start - entity_2_end
        distance_2 = entity_2_start - entity_1_end
        true_char_difference = max(distance_1,distance_2)

        word_dict = {}
        sentence_words = self.sentence.sentence_words
        for word_position in range(len(sentence_words)):
            if sentence_words[word_position] not in word_dict:
                word_dict[sentence_words[word_position]] = []
            word_dict[sentence_words[word_position]].append(word_position)

        pairs = itertools.product(word_dict[entity_1], word_dict[entity_2])
        smallest_distance = float("inf")
        right_pairs = (-1, -1)
        for p in pairs:
            if p[0] < p[1]:
                between_length = len(''.join(sentence_words[p[0] + 1:p[1]]))
                difference = abs(true_char_difference - between_length)
                if difference < smallest_distance:
                    smallest_distance = difference
                    right_pairs = (p[0], p[1])
            else:
                between_length = len(''.join(sentence_words[p[1] + 1:p[0]]))
                difference = abs(true_char_difference - between_length)
                if difference < smallest_distance:
                    smallest_distance = difference
                    right_pairs = (p[1], p[0])

        self.between_words = sentence_words[right_pairs[0] + 1:right_pairs[1]]


    def fix_word_lists(self,entity_a_text_dict,entity_b_text_dict):
        #fix dependency_word_list

        entity_a_text = entity_a_text_dict[self.sentence.pmid + '|' + self.sentence.sentence_no]
        entity_b_text = entity_b_text_dict[self.sentence.pmid + '|' + self.sentence.sentence_no]
        self.dependency_words = list_correction(self.dependency_words,entity_a_text,entity_b_text)
        self.between_words = list_correction(self.between_words,entity_a_text,entity_b_text)

        new_elements = []
        for element in self.dependency_elements:
            elements = element.split('|')
            elements_corrected = list_correction(elements,entity_a_text,entity_b_text)
            new_elements.append('|'.join(elements_corrected))
        self.dependency_elements = new_elements




    def build_features(self, dep_dictionary, dep_word_dictionary, dep_type_word_element_dictionary, between_word_dictionary):
        dep_word_features = [0] * len(dep_word_dictionary)
        dep_features = [0] * len(dep_dictionary)
        dep_type_word_element_features = [0] * len(dep_type_word_element_dictionary)
        between_features = [0] * len(between_word_dictionary)

        dep_path_feature_words = set(dep_word_dictionary.keys())
        intersection_set = dep_path_feature_words.intersection(set(self.dependency_words))
        for i in intersection_set:
            dep_word_features[dep_word_dictionary[i]] = 1

        dep_type_word_element_feature_words = set(dep_type_word_element_dictionary.keys())
        intersection_set = dep_type_word_element_feature_words.intersection(set(self.dependency_elements))
        for i in intersection_set:
            dep_type_word_element_features[dep_type_word_element_dictionary[i]] = 1

        between_feature_words = set(between_word_dictionary.keys())
        between_intersection_set = between_feature_words.intersection(set(self.between_words))
        for i in between_intersection_set:
            between_features[between_word_dictionary[i]] = 1

        dep_path_string = self.dependency_path
        if dep_path_string in dep_dictionary:
            dep_features[dep_dictionary[dep_path_string]] = 1

        self.features = dep_features + dep_word_features + dep_type_word_element_features + between_features


