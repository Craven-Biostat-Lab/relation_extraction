import os
import sys
import shutil
import numpy as np
import itertools
import load_data
import time

from machine_learning_models import tf_sess_neural_network as snn
from machine_learning_models import tf_lstm as lstm
from sklearn import metrics

def write_cv_output(filename, predicts, instances,key_order):
    for k in range(len(key_order)):
        key = key_order[k]
        labels = []
        file = open(filename+'_'+key,'w')
        #file.write(key_order[k]+'\n')
        file.write('PMID\tE1\tE2\tClASS_LABEL\tPROBABILITY\n')
        for q in range(predicts[:,k].size):
            instance_label = instances[q].label[k]
            labels.append(instance_label)
            file.write(str(instances[q].sentence.pmid) + '\t' + str(instances[q].sentence.start_entity_id) + '\t' +str(instances[q].sentence.end_entity_id) + '\t'+str(instance_label) + '\t' + str(predicts[q,k]) + '\n')
        #labels = np.array(labels)
        #precision, recall, _ = metrics.precision_recall_curve(y_true=labels,probas_pred=predicts[:, k])
        #file.write('PRECISION\tRECALL\n')
        #for z in range(precision.size):
        #    file.write(str(precision[z]) + '\t' + str(recall[z]) + '\n')

        file.close()

    return

def k_fold_cross_validation(k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_a_text, entity_b_text,hidden_array,key_order):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]


    for i in range(len(all_chunks)):
        fold_chunks = all_chunks[:]
        fold_test_abstracts = set(fold_chunks.pop(i))
        fold_training_abstracts = set(list(itertools.chain.from_iterable(fold_chunks)))

        fold_training_forward_sentences = {}
        fold_training_reverse_sentences = {}
        fold_test_forward_sentences = {}
        fold_test_reverse_sentences = {}

        for key in forward_sentences:
            if key.split('|')[0] in fold_training_abstracts:
                fold_training_forward_sentences[key] = forward_sentences[key]
            elif key.split('|')[0] in fold_test_abstracts:
                fold_test_forward_sentences[key] = forward_sentences[key]

        for key in reverse_sentences:
            if key.split('|')[0] in fold_training_abstracts:
                fold_training_reverse_sentences[key] = reverse_sentences[key]
            elif key.split('|')[0] in fold_test_abstracts:
                fold_test_reverse_sentences[key] = reverse_sentences[key]


        fold_training_instances, \
        fold_dep_dictionary, \
        fold_dep_word_dictionary, \
        fold_dep_element_dictionary, \
        fold_between_word_dictionary = load_data.build_instances_training(fold_training_forward_sentences,
                                                                          fold_training_reverse_sentences,
                                                                          distant_interactions,
                                                                          reverse_distant_interactions,
                                                                          entity_a_text,
                                                                          entity_b_text,key_order)



        #train model
        X = []
        y = []
        for t in fold_training_instances:
            X.append(t.features)
            y.append(t.label)


        fold_train_X = np.array(X)
        fold_train_y = np.array(y)


        model_dir = './model_building_meta_data/test' + str(time.time()).replace('.','')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences,
                                                                fold_test_reverse_sentences,
                                                                fold_dep_dictionary, fold_dep_word_dictionary,
                                                                fold_dep_element_dictionary,
                                                                fold_between_word_dictionary,
                                                                distant_interactions, reverse_distant_interactions,
                                                                entity_a_text, entity_b_text,key_order)


        # group instances by pmid and build feature array
        fold_test_features = []
        fold_test_labels = []


        pmid_test_instances = {}
        for test_index in range(len(fold_test_instances)):
            fti = fold_test_instances[test_index]
            if fti.sentence.pmid not in pmid_test_instances:
                pmid_test_instances[fti.sentence.pmid] = []
            pmid_test_instances[fti.sentence.pmid].append(test_index)
            fold_test_features.append(fti.features)
            fold_test_labels.append(fti.label)

        fold_test_X = np.array(fold_test_features)
        fold_test_y = np.array(fold_test_labels)


        test_model = snn.neural_network_train(fold_train_X,
                                              fold_train_y,
                                              fold_test_X,
                                              fold_test_y,
                                              hidden_array,
                                              model_dir + '/', key_order)


        fold_test_predicted_prob = snn.neural_network_test(fold_test_X,fold_test_y,test_model)
        total_predicted_prob = total_predicted_prob + fold_test_predicted_prob.tolist()
        total_test = total_test + fold_test_y.tolist()
        total_instances = total_instances + fold_test_instances

    total_test = np.array(total_test)
    total_predicted_prob = np.array(total_predicted_prob)


    return total_predicted_prob, total_instances

def parallel_k_fold_cross_validation(batch_id, k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_a_text, entity_b_text,hidden_array,key_order,LSTM):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]

    total_test = [] #test_labels for instances
    total_predicted_prob = [] #test_probability returns for instances
    total_instances = []



    fold_chunks = all_chunks[:]
    fold_test_abstracts = set(fold_chunks.pop(batch_id))
    fold_training_abstracts = set(list(itertools.chain.from_iterable(fold_chunks)))

    fold_training_forward_sentences = {}
    fold_training_reverse_sentences = {}
    fold_test_forward_sentences = {}
    fold_test_reverse_sentences = {}

    for key in forward_sentences:
        if key.split('|')[0] in fold_training_abstracts:
            fold_training_forward_sentences[key] = forward_sentences[key]
        elif key.split('|')[0] in fold_test_abstracts:
            fold_test_forward_sentences[key] = forward_sentences[key]

    for key in reverse_sentences:
        if key.split('|')[0] in fold_training_abstracts:
            fold_training_reverse_sentences[key] = reverse_sentences[key]
        elif key.split('|')[0] in fold_test_abstracts:
            fold_test_reverse_sentences[key] = reverse_sentences[key]



    if LSTM is False:
        fold_training_instances, \
        fold_dep_dictionary, \
        fold_dep_word_dictionary,\
        fold_dep_element_dictionary,\
        fold_between_word_dictionary = load_data.build_instances_training(fold_training_forward_sentences,
                                                                      fold_training_reverse_sentences,
                                                                      distant_interactions,
                                                                      reverse_distant_interactions,
                                                                      entity_a_text,
                                                                      entity_b_text,key_order)


        #train model
        X = []
        y = []
        for t in fold_training_instances:
            X.append(t.features)
            y.append(t.label)


        fold_train_X = np.array(X)
        fold_train_y = np.array(y)


        model_dir = './model_building_meta_data/test' +str(batch_id) + str(time.time()).replace('.','')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences,
                                                            fold_test_reverse_sentences,
                                                            fold_dep_dictionary, fold_dep_word_dictionary,
                                                            fold_dep_element_dictionary,
                                                            fold_between_word_dictionary,
                                                            distant_interactions, reverse_distant_interactions,
                                                            entity_a_text, entity_b_text,key_order)

        # group instances by pmid and build feature array
        fold_test_features = []
        fold_test_labels = []
        pmid_test_instances = {}
        for test_index in range(len(fold_test_instances)):
            fti = fold_test_instances[test_index]
            if fti.sentence.pmid not in pmid_test_instances:
                pmid_test_instances[fti.sentence.pmid] = []
            pmid_test_instances[fti.sentence.pmid].append(test_index)
            fold_test_features.append(fti.features)
            fold_test_labels.append(fti.label)

        fold_test_X = np.array(fold_test_features)
        fold_test_y = np.array(fold_test_labels)


        test_model = snn.neural_network_train(fold_train_X,
                                          fold_train_y,
                                          fold_test_X,
                                          fold_test_y,
                                          hidden_array,
                                          model_dir + '/', key_order)


        fold_test_predicted_prob = snn.neural_network_test(fold_test_X,fold_test_y,test_model)

        total_predicted_prob = fold_test_predicted_prob.tolist()
        total_test = fold_test_y.tolist()
        total_instances = fold_test_instances

        total_test = np.array(total_test)
        total_predicted_prob = np.array(total_predicted_prob)

        return total_predicted_prob, total_instances

    else:
        fold_training_instances, \
        fold_dep_path_list_dictionary, \
        fold_dep_word_dictionary,word2vec_embeddings = load_data.build_instances_training(fold_training_forward_sentences,
                                                                          fold_training_reverse_sentences,
                                                                          distant_interactions,
                                                                          reverse_distant_interactions,
                                                                          entity_a_text,
                                                                          entity_b_text, key_order,True)

        dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_lstm_arrays(
            fold_training_instances)

        features = [dep_path_list_features,dep_word_features,dep_type_path_length,dep_word_path_length]

        model_dir = './model_building_meta_data/test' + str(batch_id) + str(time.time()).replace('.', '')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        trained_model_path = lstm.lstm_train(features,
                                             labels, len(fold_dep_path_list_dictionary), len(fold_dep_word_dictionary),
                                             model_dir + '/', key_order,word2vec_embeddings)

        fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences,
                                                                fold_test_reverse_sentences,
                                                                None, fold_dep_word_dictionary,
                                                                None,
                                                                None,
                                                                distant_interactions, reverse_distant_interactions,
                                                                entity_a_text, entity_b_text, key_order,fold_dep_path_list_dictionary)

        # group instances by pmid and build feature array
        test_dep_path_list_features, test_dep_word_features, test_dep_type_path_length, test_dep_word_path_length, test_labels = load_data.build_lstm_arrays(
            fold_test_instances)

        test_features = [test_dep_path_list_features, test_dep_word_features,test_dep_type_path_length,
                                                  test_dep_word_path_length]
        print(trained_model_path)
        fold_test_predicted_prob, fold_test_labels = lstm.lstm_test(test_features,test_labels,trained_model_path)

        assert(np.array_equal(fold_test_labels,test_labels))


        total_predicted_prob = fold_test_predicted_prob.tolist()
        total_test = fold_test_labels.tolist()
        total_instances = fold_test_instances

        total_test = np.array(total_test)
        total_predicted_prob = np.array(total_predicted_prob)

        return total_predicted_prob, total_instances
        #instance level grouping

