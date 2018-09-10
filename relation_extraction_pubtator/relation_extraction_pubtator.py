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
from machine_learning_models import tf_lstm as lstm

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics



def write_output(filename, predicts, instances, key_order):
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

def predict_sentences_lstm(model_file, pubtator_file, entity_a, entity_b):
    """
    predict instances using LSTM
    :param model_file:
    :param pubtator_file:
    :param entity_a:
    :param entity_b:
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



    dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_lstm_arrays(predict_instances)
    predict_features = [dep_path_list_features,dep_word_features,dep_type_path_length,dep_word_path_length]

    predicted_prob = lstm.lstm_predict(predict_features,labels,model_file + '/')

    return predict_instances,predicted_prob,key_order

def predict_sentences(model_file, pubtator_file, entity_a, entity_b):

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




    predict_features = []
    predict_labels = []
    for predict_index in range(len(predict_instances)):
        pi = predict_instances[predict_index]
        predict_features.append(pi.features)
        predict_labels.append(pi.label)

    predict_features = np.array(predict_features)
    predict_labels = np.array(predict_labels)

    predicted_prob = ffnn.neural_network_predict(predict_features, predict_labels,model_file + '/')

    return predict_instances,predicted_prob,key_order


def parallel_train(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b,batch_id,LSTM):

    print(LSTM)

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
    hidden_array = []

    #k-cross val
    instance_predicts,single_instances = cv.parallel_k_fold_cross_validation(batch_id, 10, training_pmids,
                                                                             training_forward_sentences,
                                                                             training_reverse_sentences,
                                                                             distant_interactions,
                                                                             reverse_distant_interactions,
                                                                             entity_a_text,entity_b_text,hidden_array,
                                                                             key_order,LSTM)

    write_output(model_out + '_' + str(batch_id) + '_predictions', instance_predicts, single_instances, key_order)


    return batch_id

def train_lstm(model_out, pubtator_file, directional_distant_directory, symmetric_distant_directory,
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




    dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length, labels = load_data.build_lstm_arrays(training_instances)
    features = [dep_path_list_features, dep_word_features, dep_type_path_length, dep_word_path_length]

    if os.path.exists(model_out):
        shutil.rmtree(model_out)

    
    trained_model_path = lstm.lstm_train(features,labels,len(dep_path_list_dictionary),len(dep_word_dictionary),model_out + '/', key_order,word2vec_embeddings)
    


    pickle.dump([dep_path_list_dictionary, dep_word_dictionary,key_order], open(model_out + 'a.pickle','wb'))
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

    elif mode.upper() == "TRAIN_LSTM":
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

        trained_model_path = train_lstm(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b)

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
        LSTM = sys.argv[12]
        LSTM = LSTM == 'True'


        trained_model_batch = parallel_train(model_out, pubtator_file, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b,batch_id,LSTM)

        print('finished training: ' + str(trained_model_batch))

    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]
        LSTM = sys.argv[7]
        LSTM =LSTM == 'True'

        if LSTM is False:
            prediction_instances, predict_probs,key_order = predict_sentences(model_file, sentence_file, entity_a, entity_b)
            #print(total_group_instance_results)

        else:
            prediction_instances,predict_probs,key_order = predict_sentences_lstm(model_file,sentence_file,entity_a,entity_b)

        print(predict_probs)
        for key_index in range(len(key_order)):
            key = key_order[key_index]
            outfile = open(out_pairs_file + '_' + key, 'w')
            outfile.write('PMID\tENTITY_1\tENTITY_2\tCLASS_LABEL\tPROBABILITY\tSENTENCE\n')
            for i in range(len(prediction_instances)):
                pi = prediction_instances[i]
                #print(i)
                #print(' '.join(pi.sentence.sentence_words))

                outfile.write(str(pi.sentence.pmid) + '\t'
                              + str(pi.sentence.start_entity_id) + '\t'
                              + str(pi.sentence.end_entity_id) + '\t'
                              + str(pi.label[key_index]) + '\t'
                              + str(predict_probs[i,key_index])+'\t'
                              + ' '.join(pi.sentence.sentence_words))

            outfile.close()

    else:
        print("usage error")


if __name__ == "__main__":
    main()

