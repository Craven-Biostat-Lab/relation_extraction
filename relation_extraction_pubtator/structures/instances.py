import itertools

class Instance(object):

    def __init__(self,sentence,label):
        self.sentence = sentence
        self.between_words = []
        self.build_words_between_features()
        self.dependency_elements = sentence.dep_parse
        self.dependency_path =  ''
        self.dependency_words = []
        self.build_dep_path_and_words()

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


    def fix_word_lists(self,entity_a_text,entity_b_text):
        #fix dependency_word_list
        for i in range(len(self.dependency_words)):
            word = self.dependency_words[i]
            if len(entity_a_text) < len(entity_b_text):
                if word in entity_a_text:
                    self.dependency_words[i] = 'Entity_A'
                elif word in entity_b_text:
                    self.dependency_words[i] = 'Entity_B'
            else:
                if word in entity_b_text:
                    self.dependency_words[i] = 'Entity_B'
                elif word in entity_a_text:
                    self.dependency_words[i] = 'Entity_A'

        for i in range(len(self.between_words)):
            word = self.between_words[i]
            if len(entity_a_text) < len(entity_b_text):
                if word in entity_a_text:
                    self.between_words[i] = 'Entity_A'
                elif word in entity_b_text:
                    self.between_words[i] = 'Entity_B'
            else:
                if word in entity_b_text:
                    self.between_words[i] = 'Entity_B'
                elif word in entity_a_text:
                    self.between_words[i] = 'Entity_A'


