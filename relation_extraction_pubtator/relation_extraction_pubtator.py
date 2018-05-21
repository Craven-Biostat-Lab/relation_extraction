import sys
import os
import load_data
import cross_validation as cv
import numpy as np
import itertools
import collections
import shutil
import pickle
import time

from machine_learning_models import tf_neural_network as ann
from machine_learning_models import tf_sess_neural_network as snn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics


def predict_sentences(model_file, pubtator_file, entity_a, entity_b):

    predict_pmids, \
    predict_forward_sentences,\
    predict_reverse_sentences,\
    entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(pubtator_file,entity_a,entity_b)

    dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))

    predict_instances = load_data.build_instances_predict(predict_forward_sentences, predict_reverse_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_a_text,entity_b_text,key_order)

    instance_to_group_dict, group_to_instance_dict, instance_dict = cv.create_instance_groupings(predict_instances, range(len(predict_instances)))



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


def parallel_train(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,batch_id):

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
    hidden_array = [256]

    #k-cross val
    instance_predicts,single_instances = cv.parallel_k_fold_cross_validation(batch_id, 10, training_pmids,
                                                                             training_forward_sentences,
                                                                             training_reverse_sentences,
                                                                             distant_interactions,
                                                                             reverse_distant_interactions,
                                                                             entity_a_text,entity_b_text,hidden_array,
                                                                             key_order)

    cv.write_cv_output(model_out + '_' +str(batch_id)+'_instance_data.txt',instance_predicts,single_instances,key_order)


    return batch_id

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
    hidden_array = [256]

    #k-cross val
    instance_predicts, single_instances= cv.k_fold_cross_validation(10,training_pmids,training_forward_sentences,
                                                                    training_reverse_sentences,distant_interactions,
                                                                    reverse_distant_interactions,entity_a_text,
                                                                    entity_b_text,hidden_array,key_order)

    cv.write_cv_output(model_out+'_instance_data.txt',instance_predicts,single_instances,key_order)



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

    elif mode.upper() == "PARALLEL_TRAIN":
        model_out = sys.argv[2]  # location of where model should be saved after training
        pubtator_file = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        distant_rel_col = int(sys.argv[8])  # relation column
        entity_a = sys.argv[9].upper()  # entity_a
        entity_b = sys.argv[10].upper()  # entity_b
        batch_id = int(sys.argv[11]) #batch to run

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        trained_model_batch = parallel_train(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b,batch_id)

        print('finished training: ' + str(trained_model_batch))

    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]

        total_group_instances, total_group_instance_results, total_group_pmids, total_group_noisy_or,key_order = predict_sentences(model_file, sentence_file, entity_a, entity_b)
        #print(total_group_instance_results)

        outfile = open(out_pairs_file,'w')
        outfile.write('START_GENE\tSTART_GENE_SPECIES\tSTART_GENE_ENTREZ\tEND_GENE\tEND_GENE_SPECIES\tEND_GENE_ENTREZ\tPMIDS\t'+'\t'.join(key_order)+'\n')
        for i in range(len(total_group_instances)):
            pmids = set()
            for j in range(len(total_group_instances[i])):
                pmids.add(total_group_instances[i][j].sentence.pmid)
            outfile.write(total_group_instances[i][0].sentence.start_entity_raw_string +
                          '\t' + total_group_instances[i][0].sentence.start_entity_species +
                          '\t' + total_group_instances[i][0].sentence.start_entity_id +
                          '\t'+ total_group_instances[i][0].sentence.end_entity_raw_string +
                          '\t' + total_group_instances[i][0].sentence.end_entity_species +
                          '\t' + total_group_instances[i][0].sentence.end_entity_id +
                          '\tpmids:' + '|'.join(pmids) +
                          '\t' + '\t'.join(str(noise) for noise in total_group_noisy_or[i])+'\n')

        outfile.close()

    else:
        print("usage error")


if __name__ == "__main__":
    main()