import collections
import os
import itertools
import cPickle as pickle
import numpy as np

from machine_learning_models import tf_lstm as lstm
from structures.sentences import Sentence
from structures.instances import Instance
from collections import Counter

def build_dataset(words, occur_count = None):
    """Process raw mentions of features into dictionary and count dictionary"""
    num_total_words = len(set(words))
    discard_count = 0
    if occur_count is not None:
        word_count_dict = collections.Counter(words)
        discard_count = sum(1 for i in word_count_dict.values() if i < occur_count)
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


def feature_pruning(feature_dict,feature_count_tuples,prune_val):
    """Feature pruning if not done earlier - Don't really need this  function"""
    feature_count_dict = dict(feature_count_tuples)
    for key, value in feature_count_dict.iteritems():
        if value < prune_val:
            popped_element = feature_dict.pop(key)

    return feature_dict



def build_instances_predict(predict_forward_sentences, predict_reverse_sentences, dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_a_text,entity_b_text,key_order,dep_path_list_dictionary=None):
    """Builds the instances for the predict function"""
    predict_instances = []
    for key in predict_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' + splitkey[1] + '|' + splitkey[3] + '|' + splitkey[2]
        if reverse_key in predict_reverse_sentences:
            forward_predict_instance = Instance(predict_forward_sentences[key], [-1]*len(key_order))
            forward_predict_instance.fix_word_lists(entity_a_text, entity_b_text)
            reverse_predict_instance = Instance(predict_reverse_sentences[reverse_key], [0]*len(key_order))
            reverse_predict_instance.fix_word_lists(entity_a_text, entity_b_text)


            predict_instances.append(forward_predict_instance)
            predict_instances.append(reverse_predict_instance)

        else:
            continue

    if dep_path_list_dictionary is False:
        for instance in predict_instances:
            instance.build_features(dep_dictionary, dep_word_dictionary, dep_element_dictionary,  between_word_dictionary)

    else:
        for instance in predict_instances:
            instance.build_features_lstm(dep_path_list_dictionary,dep_word_dictionary)
    return predict_instances


def build_instances_testing(test_forward_sentences, test_reverse_sentences,dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, entity_a_text, entity_b_text,key_order,dep_path_type_list_dictionary=None):

    test_instances = []

    for key in test_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' + splitkey[1] + '|' + splitkey[3] + '|' + splitkey[2]
        if reverse_key in test_reverse_sentences:
            forward_test_instance = Instance(test_forward_sentences[key], [0]*len(key_order))
            forward_test_instance.fix_word_lists(entity_a_text, entity_b_text)
            reverse_test_instance = Instance(test_reverse_sentences[reverse_key], [0]*len(key_order))
            reverse_test_instance.fix_word_lists(entity_a_text, entity_b_text)

            entity_combo = (forward_test_instance.sentence.start_entity_id,
                                forward_test_instance.sentence.end_entity_id)

            for i in range(len(key_order)):
                distant_key = key_order[i]
                if 'SYMMETRIC' in distant_key:
                    if entity_combo in distant_interactions[distant_key] or entity_combo in reverse_distant_interactions[distant_key]:
                        forward_test_instance.set_label_i(1,i)
                        reverse_test_instance.set_label_i(1,i)
                else:
                    if entity_combo in distant_interactions[distant_key]:
                        forward_test_instance.set_label_i(1, i)
                    elif entity_combo in reverse_distant_interactions[distant_key]:
                        reverse_test_instance.set_label_i(1, i)



            test_instances.append(forward_test_instance)
            test_instances.append(reverse_test_instance)

        else:
            continue

    if dep_path_type_list_dictionary is None:
        for instance in test_instances:
            instance.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary,  between_word_dictionary)
    else:
        for instance in test_instances:
            instance.build_features_lstm(dep_path_type_list_dictionary,dep_path_word_dictionary)
    return test_instances



def build_instances_training(
        training_forward_sentences,training_reverse_sentences,distant_interactions,
        reverse_distant_interactions, entity_a_text, entity_b_text,key_order,LSTM=False):

    path_word_vocabulary = []
    words_between_entities_vocabulary = []
    dep_type_vocabulary = []
    dep_type_list_vocabulary = []
    dep_type_word_elements_vocabulary = []
    print(key_order)


    candidate_instances = []
    for key in training_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' +splitkey[1] +'|' +splitkey[3] + '|' + splitkey[2]
        if reverse_key in training_reverse_sentences:
            forward_train_instance = Instance(training_forward_sentences[key],[0]*len(key_order))
            forward_train_instance.fix_word_lists(entity_a_text,entity_b_text)
            reverse_train_instance = Instance(training_reverse_sentences[reverse_key],[0]*len(key_order))
            reverse_train_instance.fix_word_lists(entity_a_text, entity_b_text)

            entity_combo = (forward_train_instance.sentence.start_entity_id,
                             forward_train_instance.sentence.end_entity_id)


            for i in range(len(key_order)):
                distant_key = key_order[i]
                if 'SYMMETRIC' in distant_key:
                    if entity_combo in distant_interactions[distant_key] or entity_combo in reverse_distant_interactions[distant_key]:
                        forward_train_instance.set_label_i(1,i)
                        reverse_train_instance.set_label_i(1,i)
                else:
                    if entity_combo in distant_interactions[distant_key]:
                        forward_train_instance.set_label_i(1, i)
                    elif entity_combo in reverse_distant_interactions[distant_key]:
                        reverse_train_instance.set_label_i(1, i)

            path_word_vocabulary += forward_train_instance.dependency_words
            path_word_vocabulary += reverse_train_instance.dependency_words
            words_between_entities_vocabulary += forward_train_instance.between_words
            words_between_entities_vocabulary += reverse_train_instance.between_words
            dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
            dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
            dep_type_vocabulary.append(forward_train_instance.dependency_path_string)
            dep_type_vocabulary.append(reverse_train_instance.dependency_path_string)
            dep_type_list_vocabulary += forward_train_instance.dependency_path_list
            dep_type_list_vocabulary += reverse_train_instance.dependency_path_list
            candidate_instances.append(forward_train_instance)
            candidate_instances.append(reverse_train_instance)


        else:
            continue

    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary,5)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary,5)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(
        dep_type_word_elements_vocabulary,5)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(
        words_between_entities_vocabulary,5)
    dep_type_list_data, dep_type_list_count,dep_type_list_dictionary,dep_type_list_reversed_dictionary = build_dataset(dep_type_list_vocabulary,0)


    if LSTM is False:
        for ci_index in range(len(candidate_instances)):
            ci = candidate_instances[ci_index]
            ci.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

        return candidate_instances, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary
    else:
        dep_type_list_dictionary['UNKNOWN_WORD'] = len(dep_type_list_dictionary)
        dep_path_word_dictionary['UNKNOWN_WORD'] = len(dep_path_word_dictionary)
        word2vec_embeddings = None
        word2vec_path = os.path.dirname(os.path.realpath(__file__)) + '/machine_learning_models/PubMed-w2v.bin'
        print(word2vec_path)
        if os.path.exists(word2vec_path):
            print('embeddings exist')
            word2vec_words, word2vec_vectors,dep_path_word_dictionary = lstm.load_bin_vec(word2vec_path)
            #dep_path_word_dictionary = {k: v for v, k in enumerate(word2vec_words)}
            word2vec_embeddings = np.array(word2vec_vectors)
            print('finished fetching embeddings and placing in dictionary')

        for ci_index in range(len(candidate_instances)):
            ci = candidate_instances[ci_index]
            ci.build_features_lstm(dep_type_list_dictionary,dep_path_word_dictionary)

        return candidate_instances,dep_type_list_dictionary,dep_path_word_dictionary,word2vec_embeddings

def load_gene_gene_abstract_sentences(pubtator_file, entity_a_species, entity_b_species, entity_a_set, entity_b_set):
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
            start_entity_text = l[2]
            start_entity_loc = l[3]
            end_entity_text = l[4]
            end_entity_loc = l[5]
            start_entity_raw_string = l[6]
            end_entity_raw_string = l[7]
            start_entity_full_norm = l[8]
            end_entity_full_norm = l[9]
            start_entity_type = l[10]
            end_entity_type = l[11]
            dep_parse = l[12].split(' ')
            sentence = l[13].split(' ')

            start_entity_norm_split = start_entity_full_norm.split('(Tax:')
            start_entity_id = start_entity_norm_split[0]
            start_entity_species = '9606'
            if start_entity_id in entity_a_set:
                start_entity_species = entity_a_species
            elif start_entity_id in entity_b_set:
                start_entity_species = entity_b_species
            else:
                if len(start_entity_norm_split) > 1:
                    start_entity_species = start_entity_norm_split[1][:-1]


            end_entity_norm_split = end_entity_full_norm.split('(Tax:')
            end_entity_id = end_entity_norm_split[0]
            end_entity_species = '9606'
            if end_entity_id in entity_a_set:
                end_entity_species = entity_a_species
            elif end_entity_id in entity_b_set:
                end_entity_species = entity_b_species
            else:
                if len(end_entity_norm_split) > 1:
                    end_entity_species = end_entity_norm_split[1][:-1]


            if pmid+'|'+sentence_no not in entity_a_texts:
                entity_a_texts[pmid+'|'+sentence_no] = set()
            if pmid+'|'+sentence_no not in entity_b_texts:
                entity_b_texts[pmid+'|'+sentence_no]=set()

            if entity_a_species == start_entity_species and entity_b_species == end_entity_species:
                entity_a_texts[pmid+'|'+sentence_no].add(start_entity_text)
                entity_b_texts[pmid+'|'+sentence_no].add(end_entity_text)

            if entity_a_species == end_entity_species and entity_b_species == start_entity_species:
                entity_a_texts[pmid+'|'+sentence_no].add(end_entity_text)
                entity_b_texts[pmid+'|'+sentence_no].add(start_entity_text)


            label = pmid + '|' + sentence_no + '|' + start_entity_loc + '|' + end_entity_loc

            pubtator_sentence = Sentence(pmid,sentence_no,start_entity_text,start_entity_loc,end_entity_text,end_entity_loc,
                                          start_entity_raw_string,end_entity_raw_string,start_entity_full_norm,end_entity_full_norm,start_entity_type,
                                         end_entity_type, start_entity_id,end_entity_id,
                                         start_entity_species, end_entity_species,dep_parse, sentence)



            if start_entity_type.upper() == 'GENE' and end_entity_type.upper() == 'GENE' and entity_a_species != entity_b_species:
                pmid_list.add(pmid)

                if entity_a_species == pubtator_sentence.start_entity_species and entity_b_species == pubtator_sentence.end_entity_species:
                    forward_sentences[label] = pubtator_sentence


                elif entity_a_species == pubtator_sentence.end_entity_species and entity_b_species == pubtator_sentence.start_entity_species:
                    reverse_sentences[label] = pubtator_sentence

                else:
                    continue

            elif start_entity_type.upper() == 'GENE' and end_entity_type.upper() == 'GENE' and entity_a_species == entity_b_species:
                pmid_list.add(pmid)
                same_species = entity_a_species

                reverse_label = pmid + '|' + sentence_no + '|' + end_entity_loc + '|' + start_entity_loc

                if pubtator_sentence.start_entity_species == same_species and pubtator_sentence.end_entity_species == same_species:
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
        entity_a_set = load_entity_set(os.path.dirname(os.path.realpath(__file__)) +'/entity_ids/gene_ids/'+entity_a_specific +'.txt',2)
        entity_b_set = load_entity_set(os.path.dirname(os.path.realpath(__file__)) +'/entity_ids/gene_ids/'+entity_b_specific +'.txt',2)
        return load_gene_gene_abstract_sentences(pubtator_file, entity_a_specific, entity_b_specific, entity_a_set, entity_b_set)


def load_distant_kb(distant_kb_file, column_a, column_b,distant_rel_col):
    '''Loads data from knowledge base into tuples'''
    distant_interactions = set()
    reverse_distant_interactions = set()
    #reads in lines from kb file
    file = open(distant_kb_file,'rU')
    #print(distant_kb_file)
    lines = file.readlines()
    file.close()
    for l in lines:
        split_line = l.split('\t')
        #column_a is entity_1 column_b is entity 2
        #print(split_line)
        tuple = (split_line[column_a],split_line[column_b])
        if split_line[distant_rel_col].endswith('by') is False:
            distant_interactions.add(tuple)
        else:
            reverse_distant_interactions.add(tuple)

    #returns both forward and backward tuples for relations
    return distant_interactions,reverse_distant_interactions

def load_distant_directories(directional_distant_directory,symmetric_distant_directory,distant_entity_a_col,
                             distant_entity_b_col,distant_rel_col):
    forward_dictionary = {}
    reverse_dictionary = {}
    for filename in os.listdir(directional_distant_directory):
        if filename.endswith('.txt') is False:
            continue
        distant_interactions,reverse_distant_interactions = load_distant_kb(directional_distant_directory+'/'+filename,
                                                                            distant_entity_a_col,distant_entity_b_col,distant_rel_col)
        forward_dictionary[filename] = distant_interactions
        reverse_dictionary[filename] = reverse_distant_interactions

    for filename in os.listdir(symmetric_distant_directory):
        if filename.endswith('.txt') is False:
            continue
        distant_interactions,reverse_distant_interactions = load_distant_kb(symmetric_distant_directory+'/'+filename,
                                                                            distant_entity_a_col,distant_entity_b_col,distant_rel_col)
        forward_dictionary['SYMMETRIC'+filename] = distant_interactions
        reverse_dictionary['SYMMETRIC'+filename] = reverse_distant_interactions

    return forward_dictionary,reverse_dictionary

def load_entity_set(filename,column):
    entity_set = set()
    if os.path.isfile(filename) is False:
        return entity_set
    with open(filename) as file:
        for line in file:
            splitline = line.split('\t')
            entity_set.add(splitline[column])

    return entity_set


def build_lstm_arrays(instances):
    dep_path_list_features = []
    dep_word_features = []
    dep_type_path_length = []
    dep_word_path_length = []
    labels = []
    instance_sentences = set()
    for i in range(len(instances)):
        t = instances[i]
        # instance_sentences.add(' '.join(t.sentence.sentence_words))
        dep_path_list_features.append(t.features[0:100])
        dep_word_features.append(t.features[100:200])
        dep_type_path_length.append(t.features[200])
        dep_word_path_length.append(t.features[201])
        labels.append(t.label)

    dep_path_list_features = np.array(dep_path_list_features)
    dep_word_features = np.array(dep_word_features)
    dep_type_path_length = np.array(dep_type_path_length)
    dep_word_path_length = np.array(dep_word_path_length)
    labels = np.array(labels)

    return dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels