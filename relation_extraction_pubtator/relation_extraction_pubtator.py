import sys
import os

import load_data

def distant_train(model_out,pubtator_file,distant_file ,distant_e1_col,distant_e2_col,distant_rel_col,entity_1, entity_2, symmetric):

    print(distant_file)
    print(distant_e1_col)
    print(distant_e2_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_kb(distant_file, distant_e1_col,
                                                                                   distant_e2_col, distant_rel_col)


    training_pmids,training_forward_sentences,training_reverse_sentences, entity_1_text, entity_2_text = load_data.load_pubtator_abstract_sentences(
        pubtator_file,entity_1,entity_2)

    print(entity_1_text.intersection(entity_2_text))
    print(entity_1_text)
    print(entity_2_text)
    total_training_forward_sentences = {}
    total_training_reverse_sentences = {}

    for key in training_forward_sentences:
        if key.split('|')[0] in training_pmids:
            total_training_forward_sentences[key] = training_forward_sentences[key]

    for key in training_reverse_sentences:
        if key.split('|')[0] in training_pmids:
            total_training_reverse_sentences[key] = training_reverse_sentences[key]

    training_instances, dep_dictionary, dep_word_dictionary, element_dictionary, between_word_dictionary = load_data.build_instances_training(
        total_training_forward_sentences,total_training_reverse_sentences,distant_interactions,
        reverse_distant_interactions, entity_1_text, entity_2_text, symmetric)

    print(len(training_instances))
    count = 0
    for t in training_instances:
        if t.get_label() == 1:
            count +=1

    print(len(training_instances[0].features))
    print(len(training_instances[1].features))
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

    else:
        print("usage error")


if __name__ == "__main__":
    main()