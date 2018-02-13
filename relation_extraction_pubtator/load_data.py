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
        word_count_dict = collections.Counter(words)
        print(len(word_count_dict))
        discard_count = sum(1 for i in word_count_dict.values() if i < occur_count)
        print(discard_count)
    num_words = num_total_words - discard_count
    print(num_words)
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

def feature_pruning(feature_dict,feature_count_tuples,prune_val):
    feature_count_dict = dict(feature_count_tuples)
    for key, value in feature_count_dict.iteritems():
        if value < prune_val:
            popped_element = feature_dict.pop(key)

    return feature_dict



def build_instances_predict(predict_forward_sentences, predict_reverse_sentences, dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_a_text,entity_b_text, symmetric):
    predict_instances = []
    for key in predict_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' + splitkey[1] + '|' + splitkey[3] + '|' + splitkey[2]
        if reverse_key in predict_reverse_sentences:
            forward_predict_instance = Instance(predict_forward_sentences[key], -1)
            forward_predict_instance.fix_word_lists(entity_a_text, entity_b_text)
            reverse_predict_instance = Instance(predict_reverse_sentences[reverse_key], -1)
            reverse_predict_instance.fix_word_lists(entity_a_text, entity_b_text)

            if symmetric is False:
                predict_instances.append(forward_predict_instance)
                predict_instances.append(reverse_predict_instance)

            else:
                forward_dep_type_path = ' '.join(forward_predict_instance.dependency_path)
                reverse_dep_type_path = ' '.join(reverse_predict_instance.dependency_path)

                if forward_dep_type_path in dep_dictionary:
                    predict_instances.append(forward_predict_instance)
                elif reverse_dep_type_path in dep_dictionary:
                    predict_instances.append(reverse_predict_instance)
                else:
                    predict_instances.append(forward_predict_instance)
        else:
            continue

    for instance in predict_instances:
        instance.build_features(dep_dictionary, dep_word_dictionary, dep_element_dictionary,  between_word_dictionary)

    return predict_instances


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

            entity_combo = (forward_test_instance.sentence.entity_1_simple_norm,
                                forward_test_instance.sentence.entity_2_simple_norm)


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

            entity_combo = (forward_train_instance.sentence.entity_1_simple_norm,
                             forward_train_instance.sentence.entity_2_simple_norm)


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

    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary,5)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary,5)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(
        dep_type_word_elements_vocabulary,5)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(
        words_between_entities_vocabulary,5)


    #print(dep_dictionary)
    #print(dep_path_word_dictionary)
    #print(between_word_dictionary)
    #print(dep_element_dictionary)

    for ci in candidate_instances:
        ci.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return candidate_instances, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary






def load_gene_gene_abstract_sentences(pubtator_file, entity_a_species, entity_b_species):
    entity_a_texts = {}
    entity_b_texts = {}
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

            e1_split = entity_1_norm.split('(Tax:')
            entity_1_norm_simple = e1_split[0]
            entity_1_species = 'HUMAN'
            if len(e1_split) > 1:
                entity_1_species = e1_split[1][:-1]
            e2_split = entity_2_norm.split('(Tax:')
            entity_2_norm_simple = e2_split[0]
            entity_2_species = 'HUMAN'
            if len(e2_split) > 1:
                entity_2_species = e2_split[1][:-1]


            if pmid not in entity_a_texts:
                entity_a_texts[pmid+'|'+sentence_no] = set()
            if pmid not in entity_b_texts:
                entity_b_texts[pmid+'|'+sentence_no]=set()

            if entity_a_species == entity_1_species and entity_b_species == entity_2_species:
                entity_a_texts[pmid+'|'+sentence_no].add(entity_1_text)
                entity_b_texts[pmid+'|'+sentence_no].add(entity_2_text)

            if entity_a_species == entity_2_species and entity_b_species == entity_1_species:
                entity_a_texts[pmid+'|'+sentence_no].add(entity_2_text)
                entity_b_texts[pmid+'|'+sentence_no].add(entity_1_text)


            label = pmid + '|' + sentence_no + '|' + entity_1_loc + '|' + entity_2_loc
            pubtator_sentence = Sentence(pmid,sentence_no,entity_1_text,entity_1_loc,entity_2_text,entity_2_loc,
                                          entity_1_formal,entity_2_formal,entity_1_norm,entity_2_norm,entity_1_type,
                                         entity_2_type, entity_1_norm_simple,entity_2_norm_simple,
                                         entity_1_species, entity_2_species,dep_parse, sentence)


            if entity_1_type.upper() == 'GENE' and entity_2_type.upper() == 'GENE' and entity_a_species != entity_b_species:
                pmid_list.add(pmid)

                if entity_a_species == pubtator_sentence.entity_1_species and entity_b_species == pubtator_sentence.entity_2_species:
                    forward_sentences[label] = pubtator_sentence


                elif entity_a_species == pubtator_sentence.entity_2_species and entity_b_species == pubtator_sentence.entity_1_species:
                    reverse_sentences[label] = pubtator_sentence

                else:
                    continue

            elif entity_1_type.upper() == 'GENE' and entity_2_type.upper() == 'GENE' and entity_a_species == entity_b_species:
                pmid_list.add(pmid)
                same_species = entity_a_species

                reverse_label = pmid + '|' + sentence_no + '|' + entity_2_loc + '|' + entity_1_loc

                if pubtator_sentence.entity_1_species == same_species and pubtator_sentence.entity_2_species == same_species:
                    if reverse_label in forward_sentences:
                        reverse_sentences[label] = pubtator_sentence
                    else:
                        forward_sentences[label] = pubtator_sentence

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