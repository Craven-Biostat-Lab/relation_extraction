import sys
import os
import load_data
import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt

from machine_learning_models import machine_learning_models as ml

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import metrics


def create_instance_groupings(all_instances, group_instances, symmetric):

    instance_to_group_dict = {}
    group_to_instance_dict = {}
    instance_dict = {}
    group = 0

    for i in group_instances:
        ig = all_instances[i]
        start_norm = ig.sentence.entity_1_norm
        end_norm = ig.sentence.entity_2_norm
        instance_dict[i] = [start_norm, end_norm]
        instance_to_group_dict[i] = group
        group += 1

    for i1 in group_instances:
        instance_1 = all_instances[i1]
        for i2 in group_instances:
            instance_2 = all_instances[i2]

            recent_update = False

            if instance_1 == instance_2 or instance_1.get_label() != instance_2.get_label():
                continue

            if instance_dict[i1][0] == instance_dict[i2][0] and \
                            instance_dict[i1][1] == instance_dict[i2][1]:
                instance_to_group_dict[i1] = instance_to_group_dict[i2]
                recent_update = True

            # check reverse direction if relation is symmetric and the forward direction wasn't incorporated
            if symmetric is True and recent_update is False:
                if instance_dict[i1][1] == instance_dict[i2][0]  and \
                                instance_dict[i1][0] == instance_dict[i2][1]:
                    instance_to_group_dict[i1] = instance_to_group_dict[i2]

    for i in instance_to_group_dict:
        if instance_to_group_dict[i] not in group_to_instance_dict:
            group_to_instance_dict[instance_to_group_dict[i]] = []
        group_to_instance_dict[instance_to_group_dict[i]].append(i)

    return instance_to_group_dict, group_to_instance_dict, instance_dict

def k_fold_cross_validation(k,pmids,forward_sentences,reverse_sentences, distant_interactions, reverse_distant_interactions,
                            entity_1_text, entity_2_text, symmetric):

    pmids = list(pmids)
    #split training sentences for cross validation
    ten_fold_length = len(pmids)/k
    all_chunks = [pmids[i:i + ten_fold_length] for i in xrange(0, len(pmids), ten_fold_length)]



    total_test = np.array([])
    total_predicted_prob = np.array([])
    for i in range(len(all_chunks)):
        print('Fold #: ' + str(i))
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


        fold_training_instances, fold_dep_dictionary, fold_dep_word_dictionary, fold_dep_element_dictionary, fold_between_word_dictionary = load_data.build_instances_training(
            fold_training_forward_sentences, fold_training_reverse_sentences, distant_interactions,
            reverse_distant_interactions, entity_1_text, entity_2_text, symmetric)



        #train model
        X = []
        y = []
        for t in fold_training_instances:
            X.append(t.features)
            y.append(t.label)


        fold_train_X = np.array(X)
        fold_train_y = np.array(y)

        #model = LogisticRegression()
        #model.fit(fold_train_X, fold_train_y)
        hidden_array = [10,10]
        test_model_path = ml.artificial_neural_network_train(fold_train_X,fold_train_y,hidden_array,'./model_building_meta_data/test' + str(i))



        fold_test_instances = load_data.build_instances_testing(fold_test_forward_sentences, fold_test_reverse_sentences,
                                                                fold_dep_dictionary, fold_dep_word_dictionary,
                                                                fold_dep_element_dictionary,fold_between_word_dictionary,
                                                                distant_interactions,reverse_distant_interactions,
                                                                entity_1_text,entity_2_text,symmetric)

        #group instances by pmid and build feature array
        fold_test_features = []
        fold_test_labels = []
        pmid_test_instances = {}
        for i in range(len(fold_test_instances)):
            fti = fold_test_instances[i]
            if fti.sentence.pmid not in pmid_test_instances:
                pmid_test_instances[fti.sentence.pmid] = []
            pmid_test_instances[fti.sentence.pmid].append(i)
            fold_test_features.append(fti.features)
            fold_test_labels.append(fti.label)

        fold_test_X = np.array(fold_test_features)
        fold_test_y = np.array(fold_test_labels)

        fold_test_predicted_prob = ml.artificial_neural_network_test(fold_test_X, fold_test_y, test_model_path)

        for abstract_pmid in pmid_test_instances:
            instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(fold_test_instances,
                pmid_test_instances[abstract_pmid],symmetric)

            for g in group_to_instance_dict:
                predicted_prob = []
                group_labels = []
                for test_instance_index in group_to_instance_dict[g]:
                    predicted_prob.append(fold_test_predicted_prob[test_instance_index])
                    group_labels.append(fold_test_instances[test_instance_index].label)

                group_test_y = np.unique(group_labels)
                if group_test_y.size == 1:
                    total_test = np.append(total_test, group_test_y[0])
                else:
                    continue
                    print('error')
                    # total_test = np.append(total_test,group_y)

                predicted_prob = np.array(predicted_prob)
                negation_predicted_prob = 1 - predicted_prob
                noisy_or = 1 - np.prod(negation_predicted_prob)
                total_predicted_prob = np.append(total_predicted_prob, noisy_or)


                # Generate precision recall curves

    positives = collections.Counter(total_test)[1]
    accuracy = float(positives) / total_test.size
    precision, recall, _ = metrics.precision_recall_curve(total_test, total_predicted_prob, 1)

    return precision,recall,accuracy

def predict_sentences(model_file, pubtator_file, entity_1, entity_2, symmetric,threshold):

    predict_pmids, predict_forward_sentences, predict_reverse_sentences, entity_1_text, entity_2_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file,entity_1,entity_2)

    model, dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary = joblib.load(
        model_file)

    predict_instances = load_data.build_instances_predict(predict_forward_sentences, predict_reverse_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,entity_1_text,entity_2_text, symmetric)

    instance_to_group_dict, group_to_instance_dict, instance_dict = create_instance_groupings(
        predict_instances, symmetric)

    instances = []
    pair_labels = []
    group_pmids = []
    for g in group_to_instance_dict:
        group_X = []
        group_y = []
        pmid_set = set()
        for ti in group_to_instance_dict[g]:
            instance = ti
            group_X.append(ti.features)
            pmid_set.add(ti.sentence.pmid)

        group_predict_X = np.array(group_X)



        predicted_prob = model.predict_proba(group_predict_X)[:, 1]
        negation_predicted_prob = 1 - predicted_prob
        noisy_or = 1 - np.prod(negation_predicted_prob)
        instances.append(instance)
        group_pmids.append('|'.join(list(pmid_set)))
        if noisy_or >= threshold:
            pair_labels.append(1)
        else:
            pair_labels.append(0)

    return instances, pair_labels, group_pmids
        


def distant_train(model_out,pubtator_file,distant_file ,distant_e1_col,distant_e2_col,distant_rel_col,entity_1, entity_2, symmetric):

    print(distant_file)
    print(distant_e1_col)
    print(distant_e2_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_kb(distant_file, distant_e1_col,
                                                                                   distant_e2_col, distant_rel_col)


    training_pmids,training_forward_sentences,training_reverse_sentences, entity_1_text, entity_2_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file,entity_1,entity_2)


    #k-cross val
    precision,recall, accuracy = k_fold_cross_validation(10,training_pmids,training_forward_sentences,training_reverse_sentences,distant_interactions,
                            reverse_distant_interactions,entity_1_text,entity_2_text,symmetric)


    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

    plt.plot((0.0, 1.0), (accuracy, accuracy))

    plt.title(os.path.basename(model_out).split('.')[0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

    '''
    training_instances, dep_dictionary, dep_word_dictionary, element_dictionary, between_word_dictionary = load_data.build_instances_training(
        training_forward_sentences, training_reverse_sentences,distant_interactions,
        reverse_distant_interactions, entity_1_text, entity_2_text, symmetric)

    print(len(training_instances))

    X = []
    y = []
    instance_sentences = set()
    for t in training_instances:
        instance_sentences.add(' '.join(t.sentence.sentence_words))
        X.append(t.features)
        y.append(t.label)

    X_train = np.array(X)
    y_train = np.ravel(y)


    trained_model_path = ml.artificial_neural_network_train(X_train,y_train,model_out)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print('Number of Sentences')
    print(len(instance_sentences))
    print('Number of Instances')
    print(len(training_instances))
    print('Number of Positive Instances')
    print(y.count(1))
    print(model.get_params)
    print('Number of dependency paths ')
    print(len(dep_dictionary))
    print('Number of dependency words')
    print(len(dep_word_dictionary))
    print('Number of between words')
    print(len(between_word_dictionary))
    print('Number of elements')
    print(len(element_dictionary))
    print('length of feature space')
    print(len(dep_dictionary) + len(dep_word_dictionary) + len(element_dictionary) + len(between_word_dictionary))
    joblib.dump((model, dep_dictionary, dep_word_dictionary, element_dictionary, between_word_dictionary), model_out)
    '''
    print("trained model")



def main():
    ''' Main method, mode determines whether program runs training, testing, or prediction'''
    mode = sys.argv[1]  # what option
    if mode.upper() == "DISTANT_TRAIN":
        model_out = sys.argv[2]  # location of where model should be saved after training
        pubtator_file = sys.argv[3]  # xml file of sentences from Stanford Parser
        distant_file = sys.argv[4]  # distant supervision knowledge base to use
        distant_e1_col = int(sys.argv[5])  # entity 1 column
        distant_e2_col = int(sys.argv[6])  # entity 2 column
        distant_rel_col = int(sys.argv[7])  # relation column
        entity_1 = sys.argv[8].upper()  # entity_1
        entity_2 = sys.argv[9].upper()  # entity_2
        symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        distant_train(model_out, pubtator_file, distant_file, distant_e1_col, distant_e2_col, distant_rel_col, entity_1,entity_2, symmetric)


    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_1 = sys.argv[4].upper()
        entity_2 = sys.argv[5].upper()
        symmetric = sys.argv[6].upper() in ['TRUE', 'Y', 'YES']
        threshold = float(sys.argv[7])
        out_pairs_file = sys.argv[8]

        predicted_instances, predicted_labels, group_pmids = predict_sentences(model_file, sentence_file, entity_1,
                                                                  entity_2, symmetric,threshold)

        outfile = open(out_pairs_file,'w')

        outfile.write('GENE_1\tGENE_2\tGENE_1_SPECIES\tGENE_2_SPECIES\tPUBMED_IDS\n')
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1:
                outfile.write(predicted_instances[i].sentence.entity_1_formal + '\t' + predicted_instances[i].sentence.entity_2_formal +
                              '\t' + predicted_instances[i].sentence.entity_1_species + '\t' + predicted_instances[i].sentence.entity_2_species +
                              '\t' + group_pmids[i]+'\n')

        outfile.close()




    else:
        print("usage error")


if __name__ == "__main__":
    main()