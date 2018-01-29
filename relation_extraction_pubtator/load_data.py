import collections
import itertools
import cPickle as pickle

from structures.sentences import Sentence
from structures.instances import Instance

def build_dataset(words, occur_count = None):
    """Process raw inputs into a dataset."""
    num_total_words = len(set(words))
    discard_count = 0
    if occur_count is not None:
        for c in collections.Counter(words):
            if collections.Counter(words)[c] < occur_count:
                discard_count +=1

    num_words = num_total_words - discard_count

    count = []
    count.extend(collections.Counter(words).most_common(num_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def build_instances_testing(test_forward_sentences, test_reverse_sentences,dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, entity_a_text, entity_b_text, symmetric = False):
    test_instances = []
    for key in test_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' + splitkey[1] + '|' + splitkey[3] + '|' + splitkey[2]
        if reverse_key in test_reverse_sentences:
            forward_test_instance = Instance(test_forward_sentences[key], 0)
            forward_test_instance.fix_word_lists(entity_a_text, entity_b_text)
            reverse_test_instance = Instance(test_reverse_sentences[reverse_key], 0)
            reverse_test_instance.fix_word_lists(entity_a_text, entity_b_text)

            entity_combo = (forward_test_instance.sentence.entity_1_norm.split('(')[0],
                                forward_test_instance.sentence.entity_2_norm.split('(')[0])


            if symmetric is False:
                # check if check returned true because of reverse
                if entity_combo in distant_interactions:
                    forward_test_instance.set_label(1)
                elif entity_combo in reverse_distant_interactions:
                    reverse_test_instance.set_label(1)
                else:
                    pass

                test_instances.append(forward_test_instance)
                test_instances.append(reverse_test_instance)

            #if symmetric is True
            else:
                if entity_combo in distant_interactions or \
                                entity_combo in reverse_distant_interactions:
                    forward_test_instance.set_label(1)
                    reverse_test_instance.set_label(1)

                forward_dep_type_path = ' '.join(forward_test_instance.dependency_path)
                reverse_dep_type_path = ' '.join(reverse_test_instance.dependency_path)

                if forward_dep_type_path in dep_dictionary:
                    test_instances.append(forward_test_instance)
                elif reverse_dep_type_path in dep_dictionary:
                    test_instances.append(reverse_test_instance)
                else:
                    test_instances.append(forward_test_instance)
        else:
            continue

    for instance in test_instances:
        instance.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary,  between_word_dictionary)

    return test_instances

def build_instances_training(
        training_forward_sentences,training_reverse_sentences,distant_interactions,
        reverse_distant_interactions, entity_a_text, entity_b_text, symmetric):

    path_word_vocabulary = []
    words_between_entities_vocabulary = []
    dep_type_vocabulary = []
    dep_type_word_elements_vocabulary = []

    candidate_instances = []
    for key in training_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' +splitkey[1] +'|' +splitkey[3] + '|' + splitkey[2]
        if reverse_key in training_reverse_sentences:
            forward_train_instance = Instance(training_forward_sentences[key],0)
            forward_train_instance.fix_word_lists(entity_a_text,entity_b_text)
            reverse_train_instance = Instance(training_reverse_sentences[reverse_key],0)
            reverse_train_instance.fix_word_lists(entity_a_text, entity_b_text)

            entity_combo = (forward_train_instance.sentence.entity_1_norm.split('(')[0],
                             forward_train_instance.sentence.entity_2_norm.split('(')[0])


            if symmetric is False:
                # check if check returned true because of reverse
                if entity_combo in distant_interactions:
                    path_word_vocabulary += forward_train_instance.dependency_words
                    words_between_entities_vocabulary += forward_train_instance.between_words
                    dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
                    dep_type_vocabulary.append(forward_train_instance.dependency_path)
                    forward_train_instance.set_label(1)
                    candidate_instances.append(forward_train_instance)
                elif entity_combo in reverse_distant_interactions:
                    path_word_vocabulary += reverse_train_instance.dependency_words
                    words_between_entities_vocabulary += reverse_train_instance.between_words
                    dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
                    dep_type_vocabulary.append(reverse_train_instance.dependency_path)
                    reverse_train_instance.set_label(1)
                    candidate_instances.append(reverse_train_instance)
                else:
                    path_word_vocabulary += forward_train_instance.dependency_words
                    path_word_vocabulary += reverse_train_instance.dependency_words
                    words_between_entities_vocabulary += forward_train_instance.between_words
                    words_between_entities_vocabulary += reverse_train_instance.between_words
                    dep_type_vocabulary.append(forward_train_instance.dependency_path)
                    dep_type_vocabulary.append(reverse_train_instance.dependency_path)
                    dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
                    dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
                    candidate_instances.append(forward_train_instance)
                    candidate_instances.append(reverse_train_instance)


            # if symmetric is true
            else:
                if entity_combo in distant_interactions or entity_combo in reverse_distant_interactions:

                    forward_train_instance.set_label(1)
                    reverse_train_instance.set_label(1)
                else:
                    pass
                dep_type_vocabulary_set = set(dep_type_vocabulary)
                forward_dep_type_path = forward_train_instance.dependency_path
                reverse_dep_type_path = reverse_train_instance.dependency_path

                if forward_dep_type_path in dep_type_vocabulary_set:
                    dep_type_vocabulary.append(forward_dep_type_path)
                    path_word_vocabulary += forward_train_instance.dependency_words
                    dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
                    words_between_entities_vocabulary += forward_train_instance.between_words
                    candidate_instances.append(forward_train_instance)
                elif reverse_dep_type_path in dep_type_vocabulary_set:
                    dep_type_vocabulary.append(reverse_dep_type_path)
                    path_word_vocabulary += reverse_train_instance.dependency_words
                    dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
                    words_between_entities_vocabulary += reverse_train_instance.between_words
                    candidate_instances.append(reverse_train_instance)
                else:
                    dep_type_vocabulary.append(forward_dep_type_path)
                    path_word_vocabulary += forward_train_instance.dependency_words
                    dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
                    words_between_entities_vocabulary += forward_train_instance.between_words
                    candidate_instances.append(forward_train_instance)

        else:
            continue

    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(
        dep_type_word_elements_vocabulary)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(
        words_between_entities_vocabulary)


    print(dep_dictionary)
    print(dep_path_word_dictionary)
    print(between_word_dictionary)
    print(dep_element_dictionary)

    for ci in candidate_instances:
        ci.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return candidate_instances, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary






def load_gene_gene_abstract_sentences(pubtator_file, entity_a_species, entity_b_species):
    entity_a_texts = set()
    entity_b_texts = set()
    pmid_list = set()
    forward_sentences = {}
    reverse_sentences = {}


    with open(pubtator_file) as file:
        for line in file:
            l = line.split('\t')

            # dividing each line elements of pubtator file
            pmid = l[0]
            sentence_no = l[1]
            entity_1_text = l[2]
            entity_1_loc = l[3]
            entity_2_text = l[4]
            entity_2_loc = l[5]
            entity_1_formal = l[6]
            entity_2_formal = l[7]
            entity_1_norm = l[8]
            entity_2_norm = l[9]
            entity_1_type = l[10]
            entity_2_type = l[11]
            dep_parse = l[12].split(' ')
            sentence = l[13].split(' ')

            label = pmid + '|' + sentence_no + '|' + entity_1_loc + '|' + entity_2_loc
            pubtator_sentence = Sentence(pmid,sentence_no,entity_1_text,entity_1_loc,entity_2_text,entity_2_loc,
                                          entity_1_formal,entity_2_formal,entity_1_norm,entity_2_norm,entity_1_type, entity_2_type,
                                          dep_parse, sentence)


            entity_1_correct = False
            entity_1_reverse = False
            entity_2_correct = False
            entity_2_reverse = False
            if entity_1_type.upper() == 'GENE' and entity_2_type.upper() == 'GENE':
                pmid_list.add(pmid)

                if entity_a_species != 'HUMAN':
                    if 'Tax:' + entity_a_species in entity_1_norm:
                        entity_1_correct = True
                    elif 'Tax:' + entity_a_species in entity_2_norm:
                        entity_1_reverse = True
                else:
                    if 'Tax:' not in entity_1_norm:
                        entity_1_correct = True
                    elif 'Tax:' not in entity_2_norm:
                        entity_1_reverse = True

                if entity_b_species != 'HUMAN':
                    if 'Tax:' + entity_b_species in entity_2_norm:
                        entity_2_correct = True
                    elif 'Tax:' + entity_b_species in entity_2_norm:
                        entity_2_reverse = True
                else:
                    if 'Tax:' not in entity_2_norm:
                        entity_2_correct = True
                    elif 'Tax:' not in entity_1_norm:
                        entity_2_reverse = True



                if entity_1_correct is True and entity_2_correct is True:
                    entity_a_texts.add(entity_1_text)
                    entity_b_texts.add(entity_2_text)
                    forward_sentences[label] = pubtator_sentence

                elif entity_1_reverse is True and entity_2_reverse is True:
                    entity_b_texts.add(entity_1_text)
                    entity_a_texts.add(entity_2_text)
                    reverse_sentences[label] = pubtator_sentence

                else:
                    continue


            else:
                continue

    return pmid_list, forward_sentences, reverse_sentences, entity_a_texts, entity_b_texts



def load_pubtator_abstract_sentences(pubtator_file, entity_a, entity_b):
    entity_a_elements = entity_a.split('_')
    entity_b_elements = entity_b.split('_')

    entity_a_specific = entity_a_elements[0]
    entity_b_specific = entity_b_elements[0]
    entity_a_type = entity_a_elements[1]
    entity_b_type = entity_b_elements[1]

    if entity_a_type == 'GENE' and entity_b_type == 'GENE':
        return load_gene_gene_abstract_sentences(pubtator_file, entity_a_specific, entity_b_specific)


def load_distant_kb(distant_kb_file, column_a, column_b,distant_rel_col):
    '''Loads data from knowledge base into tuples'''
    distant_interactions = set()
    reverse_distant_interactions = set()
    #reads in lines from kb file
    file = open(distant_kb_file,'rU')
    lines = file.readlines()
    file.close()
    for l in lines:
        split_line = l.split('\t')
        #column_a is entity_1 column_b is entity 2
        tuple = (split_line[column_a],split_line[column_b])
        if split_line[distant_rel_col].endswith('by') is False:
            distant_interactions.add(tuple)
        else:
            reverse_distant_interactions.add(tuple)

    #returns both forward and backward tuples for relations
    return distant_interactions,reverse_distant_interactions