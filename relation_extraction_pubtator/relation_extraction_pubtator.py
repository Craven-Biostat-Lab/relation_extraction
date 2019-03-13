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
from machine_learning_models import tf_feed_forward as ffnn
from machine_learning_models import tf_recurrent as rnn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics



def write_output(filename, instances,grads, key_order):
    '''
    Writes predictions to outfile
    :param filename: file to write predictions too
    :param predicts: prediction probabilities
    :param instances: instance structures
    :param key_order: order of relations
    :param grads: gradients of relations
    :return:
    '''

    for k in range(len(key_order)):
        key = key_order[k]
        labels = []
        file = open(filename+'_'+key,'w')
        file.write('PMID\tE1\tE2\tClASS_LABEL\tPROBABILITY\tCOS_SIM\tGROUPS\n')
        for q in range(len(instances)):
            file.write(str(instances[q].sentence.pmid) + '\t' + str(instances[q].sentence.start_entity_id) + '\t'
                       +str(instances[q].sentence.end_entity_id) + '\t'+str(grads[q][1][k]) + '\t'
                       + str(grads[q][0][k]) + '\t'+ '|'.join(map(str,grads[q][2])) + '\t' + '|'.join(map(str,grads[q][3])) + '\n')


        file.close()

    return

def predict_sentences_recurrent(model_file, pubtator_file, entity_a, entity_b):
    """
    predict instances using recurrent
    :param model_file: path of trained model
    :param pubtator_file: path of pubtator file to try
    :param entity_a: first entity value in format ENTITYID_ENTITYTYPE
    :param entity_b: second entity value
    :return:
    """

    predict_pmids, \
    predict_forward_sentences,\
    predict_reverse_sentences,\
    entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(pubtator_file,entity_a,entity_b)
    print(len(predict_forward_sentences))
    print(len(predict_reverse_sentences))

    dep_path_list_dictionary, dep_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))

    predict_instances = load_data.build_instances_predict(predict_forward_sentences, predict_reverse_sentences,None,
                                                          dep_word_dictionary, None,
                                                          None,entity_a_text,entity_b_text,key_order,dep_path_list_dictionary)



    dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_recurrent_arrays(predict_instances)
    predict_features = [dep_path_list_features,dep_word_features,dep_type_path_length,dep_word_path_length]

    predicted_prob,predict_grad = rnn.recurrent_predict(predict_features, labels, model_file + '/')

    return predict_instances,predicted_prob,predict_grad,key_order

def predict_sentences(model_file, pubtator_file, entity_a, entity_b):
    '''
    Predict sentences for feed forward neural network
    :param model_file:  path of trained model
    :param pubtator_file:  path of pubtator file
    :param entity_a: first entity value in format ENTITYID_ENTITYTYPE
    :param entity_b: second entity value
    :return:
    '''

    predict_pmids, \
    predict_forward_sentences,\
    predict_reverse_sentences,\
    entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(pubtator_file,entity_a,entity_b)

    print(len(predict_forward_sentences))
    print(len(predict_reverse_sentences))

    dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))

    predict_instances = load_data.build_instances_predict(predict_forward_sentences, predict_reverse_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_a_text,entity_b_text,key_order)

    group_instances = load_data.batch_instances(predict_instances)

    probability_dict = {}
    label_dict = {}
    cs_grad_dict = {}

    for g in group_instances:
        predict_features = []
        predict_labels = []
        for ti in group_instances[g]:
            predict_features.append(predict_instances[ti].features)
            predict_labels.append(predict_instances[ti].label)
        predict_X = np.array(predict_features)
        predict_y = np.array(predict_labels)

        predicted_prob, predict_labels, predicted_grads = ffnn.neural_network_predict(predict_X, predict_y,
                                                                                               model_file+'/')
        probability_dict[g] = predicted_prob
        label_dict[g] = predict_labels
        for i in range(len(predicted_grads)):
            print(predicted_grads[i])
            print(predicted_prob[i])
            cs_grad_dict[group_instances[g][i]] = [predicted_prob[i], predict_labels[i],
                                                   predicted_grads[i], group_instances[g]]


    return predict_instances,cs_grad_dict,key_order

def cv_train(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                   distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,recurrent):
    '''
    Train model is cross validated manner
    :param model_out: model filename
    :param pubtator_file: pubtator file for training instances
    :param directional_distant_directory:
    :param symmetric_distant_directory:
    :param distant_entity_a_col:
    :param distant_entity_b_col:
    :param distant_rel_col:
    :param entity_a:
    :param entity_b:
    :param recurrent:
    :return:
    '''
    # get distant_relations from external knowledge base file
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col,
        distant_rel_col)

    key_order = sorted(distant_interactions)
    # get pmids,sentences,
    training_pmids, training_forward_sentences, training_reverse_sentences, entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file, entity_a, entity_b)

    # hidden layer structure
    hidden_array = [256]

    # k-cross val
    single_instances, similarities,hidden_act_similarities = cv.one_fold_cross_validation(model_out,training_pmids,
                                                                              training_forward_sentences,
                                                                              training_reverse_sentences,
                                                                              distant_interactions,
                                                                              reverse_distant_interactions,
                                                                              entity_a_text, entity_b_text,
                                                                              hidden_array,
                                                                              key_order, recurrent)

    write_output(model_out + '_cv_predictions', single_instances, similarities,key_order)
    write_output(model_out + '_hidden_activations_predictions', single_instances, hidden_act_similarities, key_order)

    return True


def parallel_train(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                   distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b, batch_id, recurrent):

    print(recurrent)

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
                                                                             entity_a_text, entity_b_text, hidden_array,
                                                                             key_order, recurrent)

    write_output(model_out + '_' + str(batch_id) + '_predictions', instance_predicts, single_instances, key_order)


    return batch_id

def train_recurrent(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
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

    #training full model
    training_instances, \
    dep_path_list_dictionary, \
    dep_word_dictionary,word2vec_embeddings  = load_data.build_instances_training(training_forward_sentences,
                                                   training_reverse_sentences,
                                                   distant_interactions,
                                                   reverse_distant_interactions,
                                                   entity_a_text,
                                                   entity_b_text,
                                                   key_order,True)




    dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_recurrent_arrays(training_instances)
    features = [dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length]

    if os.path.exists(model_out):
        shutil.rmtree(model_out)

    pickle.dump([dep_path_list_dictionary, dep_word_dictionary, key_order], open(model_out + 'a.pickle', 'wb'))
    
    trained_model_path = rnn.recurrent_train(features, labels, len(dep_path_list_dictionary), len(dep_word_dictionary), model_out + '/', key_order, word2vec_embeddings)
    



    print("trained model")


    return trained_model_path

def train_labelled(model_out, pubtator_file, pubtator_labels,directional_distant_directory, symmetric_distant_directory,
                                             distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,recurrent):


    # get distant_relations from external knowledge base file
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col,
        distant_rel_col)

    key_order = sorted(distant_interactions)

    training_pmids, training_forward_sentences, training_reverse_sentences, entity_a_text, entity_b_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file, entity_a, entity_b,True)

    print(training_forward_sentences)

    hidden_array = [256]

    # k-cross val
    single_instances, gradient_similarities,hidden_act_similarities = cv.one_fold_cross_validation(model_out,training_pmids,
                                                                  training_forward_sentences,
                                                                  training_reverse_sentences,
                                                                  distant_interactions,
                                                                  reverse_distant_interactions,
                                                                  entity_a_text, entity_b_text,
                                                                  hidden_array,
                                                                  key_order, recurrent,pubtator_labels)

    write_output(model_out + '_cv_predictions', single_instances, gradient_similarities, key_order)
    write_output(model_out + '_cv_predictions_hidden_act_sim', single_instances, hidden_act_similarities, key_order)

    return True

def train_labelled_and_distant(model_out,distant_pubtator_file, labelled_pubtator_file, pubtator_labels, directional_distant_directory, symmetric_distant_directory,
                               distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b, recurrent):


    # get distant_relations from external knowledge base file
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col,
        distant_rel_col)

    key_order = sorted(distant_interactions)

    distant_pmids, distant_forward_sentences,distant_reverse_sentences,distant_entity_a_text, distant_entity_b_text = load_data.load_pubtator_abstract_sentences(
        distant_pubtator_file, entity_a, entity_b)


    labelled_pmids, labelled_forward_sentences, labelled_reverse_sentences, labelled_entity_a_text, labelled_entity_b_text = load_data.load_pubtator_abstract_sentences(
        labelled_pubtator_file, entity_a, entity_b,True)

    pmids = distant_pmids.union(labelled_pmids)

    training_forward_sentences = distant_forward_sentences.copy()
    training_forward_sentences.update(labelled_forward_sentences)

    training_reverse_sentences = distant_reverse_sentences.copy()
    training_reverse_sentences.update(labelled_reverse_sentences)

    entity_a_text = distant_entity_a_text.copy()
    for k in labelled_entity_a_text:
        if k in entity_a_text:
            entity_a_text[k] = entity_a_text[k].union(labelled_entity_a_text[k])
        else:
            entity_a_text[k] = labelled_entity_a_text[k]

    entity_b_text = distant_entity_b_text.copy()
    for k in labelled_entity_b_text:
        if k in entity_b_text:
            entity_b_text[k] = entity_b_text[k].union(labelled_entity_b_text[k])
        else:
            entity_b_text[k] = labelled_entity_b_text[k]


    print(training_forward_sentences)

    if recurrent:
        # training full model
        training_instances, \
        dep_path_list_dictionary, \
        dep_word_dictionary, word2vec_embeddings = load_data.build_instances_labelled_and_distant(training_forward_sentences,
                                                                     training_reverse_sentences, distant_interactions, reverse_distant_interactions,
                                                                     pubtator_labels,'binds',
                                                                     entity_a_text,
                                                                     entity_b_text,
                                                                     key_order,True)

        dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_recurrent_arrays(
            training_instances)
        features = [dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length]

        if os.path.exists(model_out):
            shutil.rmtree(model_out)

        trained_model_path = rnn.recurrent_train(features, labels, len(dep_path_list_dictionary),
                                                 len(dep_word_dictionary), model_out + '/', key_order,
                                                 word2vec_embeddings)

        pickle.dump([dep_path_list_dictionary, dep_word_dictionary, key_order], open(model_out + 'a.pickle', 'wb'))
        print("trained model")

        return trained_model_path
    else:

        # hidden layer structure
        hidden_array = [256]

        # k-cross val
        # instance_predicts, single_instances= cv.k_fold_cross_validation(10,training_pmids,training_forward_sentences,
        #                                                                training_reverse_sentences,distant_interactions,
        #                                                                reverse_distant_interactions,entity_a_text,
        #                                                                entity_b_text,hidden_array,key_order)

        # cv.write_cv_output(model_out+'_predictions',instance_predicts,single_instances,key_order)

        # training full model
        training_instances, \
        dep_dictionary, \
        dep_word_dictionary, \
        dep_element_dictionary, \
        between_word_dictionary = load_data.build_instances_labelled_and_distant(training_forward_sentences,
                                                                     training_reverse_sentences,distant_interactions, reverse_distant_interactions,
                                                                     pubtator_labels,'binds',
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

        print(X_train.shape)
        print(y_train.shape)



        if os.path.exists(model_out):
            shutil.rmtree(model_out)

        trained_model_path = ffnn.feed_forward_train(X_train,
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
        print(len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary) + len(
            between_word_dictionary))
        pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order],
                    open(model_out + 'a.pickle', 'wb'))
        print("trained model")

        return trained_model_path

def train_feed_forward(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
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
    #instance_predicts, single_instances= cv.k_fold_cross_validation(10,training_pmids,training_forward_sentences,
    #                                                                training_reverse_sentences,distant_interactions,
    #                                                                reverse_distant_interactions,entity_a_text,
    #                                                                entity_b_text,hidden_array,key_order)

    #cv.write_cv_output(model_out+'_predictions',instance_predicts,single_instances,key_order)



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


    trained_model_path = ffnn.feed_forward_train(X_train,
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
    if mode.upper() == "TRAIN_FEED_FORWARD":
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

        trained_model_path = train_feed_forward(model_out, pubtator_file, directional_distant_directory,
                                                symmetric_distant_directory, distant_entity_a_col, distant_entity_b_col,
                                                distant_rel_col, entity_a, entity_b)

        print(trained_model_path)

    elif mode.upper() == "TRAIN_LABELLED_AND_DISTANT":
        model_out = sys.argv[2]  # location of where model should be saved after training
        distant_pubtator_file = sys.argv[3]
        labelled_pubtator_file = sys.argv[4]  # xml file of sentences from Stanford Parser
        pubtator_labels = sys.argv[5]
        directional_distant_directory = sys.argv[6]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[7]
        distant_entity_a_col = int(sys.argv[8])  # entity 1 column
        distant_entity_b_col = int(sys.argv[9])  # entity 2 column
        distant_rel_col = int(sys.argv[10])  # relation column
        entity_a = sys.argv[11].upper()  # entity_a
        entity_b = sys.argv[12].upper()  # entity_b
        recurrent = sys.argv[13]
        recurrent = recurrent == 'True'

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        trained_model_path = train_labelled_and_distant(model_out, distant_pubtator_file,labelled_pubtator_file, pubtator_labels,directional_distant_directory, symmetric_distant_directory,
                                             distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,recurrent)

    elif mode.upper() == "TRAIN_LABELLED":
        model_out = sys.argv[2]  # location of where model should be saved after training
        pubtator_file = sys.argv[3]  # xml file of sentences from Stanford Parser
        pubtator_labels = sys.argv[4]
        directional_distant_directory = sys.argv[5]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[6]
        distant_entity_a_col = int(sys.argv[7])  # entity 1 column
        distant_entity_b_col = int(sys.argv[8])  # entity 2 column
        distant_rel_col = int(sys.argv[9])  # relation column
        entity_a = sys.argv[10].upper()  # entity_a
        entity_b = sys.argv[11].upper()  # entity_b
        recurrent = sys.argv[12]
        recurrent = recurrent == 'True'

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        trained_model_path = train_labelled(model_out, pubtator_file, pubtator_labels,directional_distant_directory, symmetric_distant_directory,
                                             distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,recurrent)

    elif mode.upper() == "TRAIN_RECURRENT":
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

        trained_model_path = train_recurrent(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                                             distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b)

    elif mode.upper() == "CV_TRAIN":
        model_out = sys.argv[2]  # location of where model should be saved after training
        pubtator_file = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        distant_rel_col = int(sys.argv[8])  # relation column
        entity_a = sys.argv[9].upper()  # entity_a
        entity_b = sys.argv[10].upper()  # entity_b
        recurrent = sys.argv[11]
        recurrent = recurrent == 'True'


        cv_train(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b,recurrent)

        print('finished training')

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
        recurrent = sys.argv[12]
        recurrent = recurrent == 'True'


        trained_model_batch = parallel_train(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b,batch_id,recurrent)

        print('finished training: ' + str(trained_model_batch))

    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]
        recurrent = sys.argv[7]
        recurrent = recurrent == 'True'
        print(recurrent)
        if recurrent is False:
            prediction_instances, predict_grad,key_order = predict_sentences(model_file, sentence_file, entity_a, entity_b)
            #print(total_group_instance_results)

        else:
            prediction_instances,predict_probs,predict_grad,key_order = predict_sentences_recurrent(model_file, sentence_file, entity_a, entity_b)


        write_output(out_pairs_file,prediction_instances,predict_grad,key_order)

    else:
        print("usage error")


if __name__ == "__main__":
    main()

