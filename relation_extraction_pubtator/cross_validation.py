import os
import shutil
import numpy as np
import itertools
import load_data
import time
import random

from machine_learning_models import tf_feed_forward as snn
from machine_learning_models import tf_recurrent as rnn
from sklearn import metrics

import tensorflow as tf


tf.contrib.summary
os.environ["CUDA_VISIBLE_DEVICES"]="1"



def cosine_sim(input_matrix):
    num_features = input_matrix.shape[1]
    input_tensor = tf.placeholder(tf.float32, [None, num_features])

    normalized = tf.nn.l2_normalize(input_tensor,axis=1)
    prod = tf.matmul(normalized,normalized,adjoint_b=True)

    dist = 1 -prod

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        output = sess.run(dist, feed_dict={input_tensor: input_matrix})

    return output

def k_fold_cross_validation(k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_a_text, entity_b_text,hidden_array,key_order,recurrent):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]

    total_predicted_prob = []
    total_test = []
    total_instances = []
    total_grad = []

    training_instances, \
    fold_dep_dictionary, \
    fold_dep_word_dictionary, \
    fold_dep_element_dictionary, \
    fold_between_word_dictionary = load_data.build_instances_training(forward_sentences,
                                                                      reverse_sentences,
                                                                      distant_interactions,
                                                                      reverse_distant_interactions,
                                                                      entity_a_text,
                                                                      entity_b_text, key_order)


    if k == 1:

        all_chunks = [1]

    for i in range(len(all_chunks)):
        if k == 1:
            testlength = int(len(pmids) * 0.2)
            random.shuffle(pmids)
            fold_test_abstracts = pmids[:testlength]
            fold_training_abstracts = pmids[testlength:]

        else:
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

        if recurrent is False:
            fold_training_instances = load_data.build_instances_testing(fold_training_forward_sentences,
                                                                        fold_training_reverse_sentences,
                                                                        fold_dep_dictionary, fold_dep_word_dictionary,
                                                                        fold_dep_element_dictionary,
                                                                        fold_between_word_dictionary,
                                                                        distant_interactions,
                                                                        reverse_distant_interactions,
                                                                        entity_a_text, entity_b_text, key_order)

            # train model
            X = []
            y = []
            for t in fold_training_instances:
                X.append(t.features)
                y.append(t.label)

            fold_train_X = np.array(X)
            fold_train_y = np.array(y)

            model_dir = os.path.dirname(os.path.realpath(__file__)) + '/model_building_meta_data/test' + str(
                i) + str(time.time()).replace('.', '')
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
            print(fold_test_y.shape)
            print(fold_test_X.shape)

            test_model = snn.feed_forward_train(fold_train_X,
                                                fold_train_y,
                                                fold_test_X,
                                                fold_test_y,
                                                hidden_array,
                                                model_dir + '/', key_order)

            fold_test_predicted_prob, fold_test_labels, fold_test_grads,fold_test_cs_grads = snn.feed_forward_test(fold_test_X, fold_test_y, test_model)
            total_predicted_prob.append(fold_test_predicted_prob)
            total_grad.append(fold_test_grads)
            total_test.append(fold_test_labels)
            total_instances = total_instances + fold_test_instances
            print('end')

        else:

            fold_training_instances, \
            fold_dep_path_list_dictionary, \
            fold_dep_word_dictionary, word2vec_embeddings = load_data.build_instances_training(
                fold_training_forward_sentences,
                fold_training_reverse_sentences,
                distant_interactions,
                reverse_distant_interactions,
                entity_a_text,
                entity_b_text, key_order, True)

            dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_recurrent_arrays(
                fold_training_instances)

            features = [dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length]

            model_dir = os.path.dirname(os.path.realpath(__file__)) + '/model_building_meta_data/test' + str(
                i) + str(time.time()).replace('.', '')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            trained_model_path = rnn.recurrent_train(features,
                                                     labels, len(fold_dep_path_list_dictionary),
                                                     len(fold_dep_word_dictionary),
                                                     model_dir + '/', key_order, word2vec_embeddings)

            fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences,
                                                                    fold_test_reverse_sentences,
                                                                    None, fold_dep_word_dictionary,
                                                                    None,
                                                                    None,
                                                                    distant_interactions, reverse_distant_interactions,
                                                                    entity_a_text, entity_b_text, key_order,
                                                                    fold_dep_path_list_dictionary)

            # group instances by pmid and build feature array
            test_dep_path_list_features, test_dep_word_features, test_dep_type_path_length, test_dep_word_path_length, test_labels = load_data.build_recurrent_arrays(
                fold_test_instances)

            test_features = [test_dep_path_list_features, test_dep_word_features, test_dep_type_path_length,
                             test_dep_word_path_length]
            print(trained_model_path)
            fold_test_predicted_prob, fold_test_labels = rnn.recurrent_test(test_features, test_labels,
                                                                            trained_model_path)

            total_predicted_prob = total_predicted_prob + fold_test_predicted_prob.tolist()
            total_test = total_test + fold_test_labels.tolist()
            total_instances = total_instances + fold_test_instances

    total_test = np.vstack(total_test)
    total_predicted_prob = np.vstack(total_predicted_prob)
    total_grad = np.vstack(total_grad)
    print('stacked')



    cs_grad = metrics.pairwise.cosine_similarity(total_grad)


    return total_predicted_prob, total_instances, cs_grad

def parallel_k_fold_cross_validation(batch_id, k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                                     entity_a_text, entity_b_text, hidden_array, key_order, recurrent):

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



    if recurrent is False:
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


        model_dir = os.path.dirname(os.path.realpath(__file__))+'/model_building_meta_data/test' +str(batch_id) + str(time.time()).replace('.','')
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


        test_model = snn.feed_forward_train(fold_train_X,
                                            fold_train_y,
                                            fold_test_X,
                                            fold_test_y,
                                            hidden_array,
                                            model_dir + '/', key_order)


        fold_test_predicted_prob = snn.feed_forward_test(fold_test_X, fold_test_y, test_model)

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

        dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_recurrent_arrays(
            fold_training_instances)

        features = [dep_path_list_features,dep_word_features,dep_type_path_length,dep_word_path_length]

        model_dir = os.path.dirname(os.path.realpath(__file__))+'/model_building_meta_data/test' + str(batch_id) + str(time.time()).replace('.', '')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        trained_model_path = rnn.recurrent_train(features,
                                                 labels, len(fold_dep_path_list_dictionary), len(fold_dep_word_dictionary),
                                                 model_dir + '/', key_order, word2vec_embeddings)

        fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences,
                                                                fold_test_reverse_sentences,
                                                                None, fold_dep_word_dictionary,
                                                                None,
                                                                None,
                                                                distant_interactions, reverse_distant_interactions,
                                                                entity_a_text, entity_b_text, key_order,fold_dep_path_list_dictionary)

        # group instances by pmid and build feature array
        test_dep_path_list_features, test_dep_word_features, test_dep_type_path_length, test_dep_word_path_length, test_labels = load_data.build_recurrent_arrays(
            fold_test_instances)

        test_features = [test_dep_path_list_features, test_dep_word_features,test_dep_type_path_length,
                                                  test_dep_word_path_length]
        print(trained_model_path)
        fold_test_predicted_prob, fold_test_labels = rnn.recurrent_test(test_features, test_labels, trained_model_path)

        assert(np.array_equal(fold_test_labels,test_labels))


        total_predicted_prob = fold_test_predicted_prob.tolist()
        total_test = fold_test_labels.tolist()
        total_instances = fold_test_instances

        total_test = np.array(total_test)
        total_predicted_prob = np.array(total_predicted_prob)

        return total_predicted_prob, total_instances
        #instance level grouping

