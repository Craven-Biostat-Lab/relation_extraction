import sys
import os
import load_data
import numpy as np
import itertools
import collections
import shutil
import pickle
import matplotlib.pyplot as plt

from machine_learning_models import tf_neural_network as ann
from machine_learning_models import tf_sess_neural_network as snn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics


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
    total_test = [] #test_labels
    total_predicted_prob = [] #test_probability returns



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


        model_dir = './model_building_meta_data/test' + str(i)
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


        for abstract_pmid in pmid_test_instances:
            instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(fold_test_instances,
                                                                                                      pmid_test_instances[abstract_pmid])

            for g in group_to_instance_dict:
                predicted_prob = []
                group_labels = []
                for test_instance_index in group_to_instance_dict[g]:
                    predicted_prob.append(fold_test_predicted_prob[test_instance_index])
                    group_labels.append(fold_test_instances[test_instance_index].label)


                predicted_prob = np.array(predicted_prob)
                negation_predicted_prob = 1 - predicted_prob
                noisy_or = 1 - np.prod(negation_predicted_prob,axis=0)
                total_predicted_prob.append(noisy_or)
                total_test.append(np.array(group_labels[0]))

    total_test = np.array(total_test)
    total_predicted_prob = np.array(total_predicted_prob)

    for k in range(len(key_order)):
        print(key_order[k])
        precision,recall,_ = metrics.precision_recall_curve(y_true=total_test[:,k],probas_pred=total_predicted_prob[:,k])
        print('PRECISION\tRECALL')
        for z in range(precision.size):
            print(str(precision[z]) + '\t' + str(recall[z]))

    return

def predict_sentences(model_file, pubtator_file, entity_a, entity_b):

    predict_pmids, \
    predict_forward_sentences,\
    predict_reverse_sentences,\
    entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(pubtator_file,entity_a,entity_b)

    dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))

    predict_instances = load_data.build_instances_predict(predict_forward_sentences, predict_reverse_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_a_text,entity_b_text,key_order)

    instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(predict_instances, range(len(predict_instances)))



    total_group_instances = []
    total_group_instance_results = []
    total_group_pmids = []
    total_group_noisy_or = []
    for g in group_to_instance_dict:
        group_X = []
        group_instances = []
        pmid_set = set()
        for predict_index in group_to_instance_dict[g]:
            pi = predict_instances[predict_index]
            group_X.append(pi.features)
            pmid_set.add(pi.sentence.pmid)
            group_instances.append(pi)

        group_predict_X = np.array(group_X)
        predicted_prob = snn.neural_network_predict(group_predict_X,model_file + '/')
        negation_predicted_prob = 1 - predicted_prob
        noisy_or = 1 - np.prod(negation_predicted_prob, axis=0)

        total_group_instances.append(group_instances)
        total_group_instance_results.append(predicted_prob)
        total_group_pmids.append(pmid_set)
        total_group_noisy_or.append(noisy_or)
    

    return total_group_instances,total_group_instance_results,total_group_pmids,total_group_noisy_or,key_order


def distant_train(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):

    #get distant_relations from external knowledge base file
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,
                                                                                            distant_rel_col)

    key_order = sorted(distant_interactions)
    #get pmids,sentences,
    training_pmids,training_forward_sentences,training_reverse_sentences, entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file,entity_a,entity_b)

    #hidden layer structure
    hidden_array = [256, 256]

    #k-cross val
    k_fold_cross_validation(10,training_pmids,training_forward_sentences,training_reverse_sentences,distant_interactions,reverse_distant_interactions,entity_a_text,entity_b_text,hidden_array,key_order)


    #training full model
    training_instances, \
    dep_dictionary, \
    dep_word_dictionary, \
    dep_element_dictionary, \
    between_word_dictionary = load_data.build_instances_training(training_forward_sentences,
                                                   training_reverse_sentences,
                                                   distant_interactions,
                                                   reverse_distant_interactions,
                                                   entity_a_text,
                                                   entity_b_text,
                                                   key_order)



    X = []
    y = []
    instance_sentences = set()
    for t in training_instances:
        instance_sentences.add(' '.join(t.sentence.sentence_words))
        X.append(t.features)
        y.append(t.label)

    X_train = np.array(X)
    y_train = np.array(y)

    if os.path.exists(model_out):
        shutil.rmtree(model_out)


    trained_model_path = snn.neural_network_train(X_train,
                                              y_train,
                                              None,
                                              None,
                                              hidden_array,
                                              model_out + '/', key_order)


    print('Number of Sentences')
    print(len(instance_sentences))
    print('Number of Instances')
    print(len(training_instances))
    print('Number of dependency paths ')
    print(len(dep_dictionary))
    print('Number of dependency words')
    print(len(dep_word_dictionary))
    print('Number of between words')
    print(len(between_word_dictionary))
    print('Number of elements')
    print(len(dep_element_dictionary))
    print('length of feature space')
    print(len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary) + len(between_word_dictionary))
    pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary,key_order], open(model_out + 'a.pickle','wb'))
    print("trained model")


    return trained_model_path


def main():
    ''' Main method, mode determines whether program runs training, testing, or prediction'''
    mode = sys.argv[1]  # what option
    if mode.upper() == "DISTANT_TRAIN":
        model_out = sys.argv[2]  # location of where model should be saved after training
        pubtator_file = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        distant_rel_col = int(sys.argv[8])  # relation column
        entity_a = sys.argv[9].upper()  # entity_a
        entity_b = sys.argv[10].upper()  # entity_b

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        trained_model_path = distant_train(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b)

        print(trained_model_path)

    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]

        total_group_instances, total_group_instance_results, total_group_pmids, total_group_noisy_or,key_order = predict_sentences(model_file, sentence_file, entity_a, entity_b)
        #print(total_group_instance_results)

        outfile = open(out_pairs_file,'w')
        outfile.write('START_GENE\tSTART_GENE_SPECIES\tSTART_GENE_ENTREZ\tEND_GENE\tEND_GENE_SPECIES\tEND_GENE_ENTREZ\t'+'\t'.join(key_order)+'\n')
        for i in range(len(total_group_instances)):
            outfile.write(total_group_instances[i][0].sentence.start_entity_raw_string +
                          '\t' + total_group_instances[i][0].sentence.start_entity_species +
                          '\t' + total_group_instances[i][0].sentence.start_entity_id +
                          '\t'+ total_group_instances[i][0].sentence.end_entity_raw_string +
                          '\t' + total_group_instances[i][0].sentence.end_entity_species +
                          '\t' + total_group_instances[i][0].sentence.end_entity_id +
                          '\t' + '\t'.join(str(noise) for noise in total_group_noisy_or[i])+'\n')

        outfile.close()




    else:
        print("usage error")


if __name__ == "__main__":
    main()