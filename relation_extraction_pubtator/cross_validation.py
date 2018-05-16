import os
import sys
import shutil
import numpy as np
import itertools
import load_data
import time

from machine_learning_models import tf_sess_neural_network as snn
from sklearn import metrics

def write_cv_output(filename,labels,predicts,key_order):
    file = open(filename,'w')
    for k in range(len(key_order)):
        file.write(key_order[k]+'\n')
        file.write('ClASS_LABEL\tPROBABILITY\n')
        for q in range(labels[:,k].size):
            file.write(str(labels[q,k]) + '\t' + str(predicts[q,k]) + '\n')
        precision, recall, _ = metrics.precision_recall_curve(y_true=labels[:, k],probas_pred=predicts[:, k])
        file.write('PRECISION\tRECALL\n')
        for z in range(precision.size):
            file.write(str(precision[z]) + '\t' + str(recall[z]) + '\n')

    file.close()


def create_instance_groupings(all_instances, group_instances):

    instance_to_group_dict = {}
    group_to_instance_dict = {}
    instance_dict = {}
    group = 0

    for i in group_instances:
        ig = all_instances[i]
        start_norm = ig.sentence.start_entity_id
        end_norm = ig.sentence.end_entity_id
        instance_dict[i] = [start_norm, end_norm]
        instance_to_group_dict[i] = group
        group += 1

    for i1 in group_instances:
        instance_1 = all_instances[i1]
        for i2 in group_instances:
            instance_2 = all_instances[i2]

            recent_update = False

            if instance_1 == instance_2:
                continue

            if instance_dict[i1][0] == instance_dict[i2][0] and \
                            instance_dict[i1][1] == instance_dict[i2][1]:
                instance_to_group_dict[i1] = instance_to_group_dict[i2]
                recent_update = True


    for i in instance_to_group_dict:
        if instance_to_group_dict[i] not in group_to_instance_dict:
            group_to_instance_dict[instance_to_group_dict[i]] = []
        group_to_instance_dict[instance_to_group_dict[i]].append(i)

    return instance_to_group_dict, group_to_instance_dict, instance_dict

def k_fold_cross_validation(k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_a_text, entity_b_text,hidden_array,key_order):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]
    total_test = [] #test_labels for instances
    total_predicted_prob = [] #test_probability returns for instances

    corpus_labels = []
    corpus_predict = []

    abstract_labels = []
    abstract_predict = []


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


        #abstract level grouping
        for abstract_pmid in pmid_test_instances:
            instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(fold_test_instances,
                                                                                                      pmid_test_instances[abstract_pmid])

            for ag in group_to_instance_dict:
                predicted_prob = []
                group_labels = []
                for a_test_instance_index in group_to_instance_dict[ag]:
                    predicted_prob.append(fold_test_predicted_prob[a_test_instance_index])
                    group_labels.append(fold_test_instances[a_test_instance_index].label)


                predicted_prob = np.array(predicted_prob)
                negation_predicted_prob = 1 - predicted_prob
                noisy_or = 1 - np.prod(negation_predicted_prob,axis=0)
                abstract_predict.append(noisy_or)
                abstract_labels.append(np.array(group_labels[0]))

        #corpus_level_ grouping
        instance_to_group_dict, group_to_instance_dict,instance_dict = create_instance_groupings(fold_test_instances,range(len(fold_test_instances)))

        for g in group_to_instance_dict:
            predicted_prob = []
            group_labels = []
            for test_instance_index in group_to_instance_dict[g]:
                predicted_prob.append(fold_test_predicted_prob[test_instance_index])
                group_labels.append(fold_test_instances[test_instance_index].label)

            predicted_prob = np.array(predicted_prob)
            negation_predicted_prob = 1 - predicted_prob
            noisy_or = 1 - np.prod(negation_predicted_prob, axis=0)
            corpus_predict.append(noisy_or)
            corpus_labels.append(np.array(group_labels[0]))

    total_test = np.array(total_test)
    total_predicted_prob = np.array(total_predicted_prob)


    abstract_predict = np.array(abstract_predict)
    abstract_labels = np.array(abstract_labels)



    corpus_predict = np.array(corpus_predict)
    corpus_labels = np.array(corpus_labels)




    return total_test,total_predicted_prob,abstract_labels,abstract_predict,corpus_labels,corpus_predict

def parallel_k_fold_cross_validation(batch_id, k, pmids, forward_sentences, reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_a_text, entity_b_text,hidden_array,key_order):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]

    total_test = [] #test_labels for instances
    total_predicted_prob = [] #test_probability returns for instances

    corpus_labels = []
    corpus_predict = []

    abstract_labels = []
    abstract_predict = []

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

    total_predicted_prob = total_predicted_prob + fold_test_predicted_prob.tolist()
    total_test = total_test + fold_test_y.tolist()

    # abstract level grouping
    for abstract_pmid in pmid_test_instances:
        instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(fold_test_instances,pmid_test_instances[abstract_pmid])

        for ag in group_to_instance_dict:
            predicted_prob = []
            group_labels = []
            for a_test_instance_index in group_to_instance_dict[ag]:
                predicted_prob.append(fold_test_predicted_prob[a_test_instance_index])
                group_labels.append(fold_test_instances[a_test_instance_index].label)

            predicted_prob = np.array(predicted_prob)
            negation_predicted_prob = 1 - predicted_prob
            noisy_or = 1 - np.prod(negation_predicted_prob, axis=0)
            abstract_predict.append(noisy_or)
            abstract_labels.append(np.array(group_labels[0]))

    # corpus_level_ grouping
    instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(fold_test_instances,range(len(fold_test_instances)))

    for g in group_to_instance_dict:
        predicted_prob = []
        group_labels = []
        for test_instance_index in group_to_instance_dict[g]:
            predicted_prob.append(fold_test_predicted_prob[test_instance_index])
            group_labels.append(fold_test_instances[test_instance_index].label)

        predicted_prob = np.array(predicted_prob)
        negation_predicted_prob = 1 - predicted_prob
        noisy_or = 1 - np.prod(negation_predicted_prob, axis=0)
        corpus_predict.append(noisy_or)
        corpus_labels.append(np.array(group_labels[0]))

    total_test = np.array(total_test)
    total_predicted_prob = np.array(total_predicted_prob)


    abstract_predict = np.array(abstract_predict)
    abstract_labels = np.array(abstract_labels)



    corpus_predict = np.array(corpus_predict)
    corpus_labels = np.array(corpus_labels)

    return total_test,total_predicted_prob,abstract_labels,abstract_predict,corpus_labels,corpus_predict
