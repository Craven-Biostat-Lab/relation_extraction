from structures.sentences import Sentence
from structures.instances import Instance

def build_instances_training(
        training_forward_sentences,training_reverse_sentences,distant_interactions,
        reverse_distant_interactions, entity_a_text, entity_b_text, symmetric):

    path_word_vocabulary = []
    words_between_entities_vocabulary = []
    dep_type_vocabulary = []
    dep_type_word_elements_vocabulary = []

    candidate_instances = []
    for key in training_forward_sentences:
        splitkey = key.split('|')
        reverse_key = splitkey[0] + '|' +splitkey[1] +'|' +splitkey[3] + '|' + splitkey[2]
        if reverse_key in training_reverse_sentences:
            forward_train_instance = Instance(training_forward_sentences[key],0)
            forward_train_instance.fix_word_lists(entity_a_text,entity_b_text)
            print(forward_train_instance.between_words)
            reverse_train_instance = Instance(training_reverse_sentences[reverse_key],0)
            reverse_train_instance.fix_word_lists(entity_a_text, entity_b_text)
            print(reverse_train_instance.between_words)







        else:
            print('key not found')
            continue


    return 0,0,0,0,0




def load_gene_gene_abstract_sentences(pubtator_file, entity_a_species, entity_b_species):
    entity_a_texts = set()
    entity_b_texts = set()
    pmid_list = set()
    forward_sentences = {}
    reverse_sentences = {}


    with open(pubtator_file) as file:
        for line in file:
            l = line.split('\t')

            # dividing each line elements of pubtator file
            pmid = l[0]
            sentence_no = l[1]
            entity_1_text = l[2]
            entity_1_loc = l[3]
            entity_2_text = l[4]
            entity_2_loc = l[5]
            entity_1_formal = l[6]
            entity_2_formal = l[7]
            entity_1_norm = l[8]
            entity_2_norm = l[9]
            entity_1_type = l[10]
            entity_2_type = l[11]
            dep_parse = l[12].split(' ')
            sentence = l[13].split(' ')

            label = pmid + '|' + sentence_no + '|' + entity_1_loc + '|' + entity_2_loc
            pubtator_sentence = Sentence(pmid,sentence_no,entity_1_text,entity_1_loc,entity_2_text,entity_2_loc,
                                          entity_1_formal,entity_2_formal,entity_1_norm,entity_2_norm,entity_1_type, entity_2_type,
                                          dep_parse, sentence)


            entity_1_correct = False
            entity_1_reverse = False
            entity_2_correct = False
            entity_2_reverse = False
            if entity_1_type.upper() == 'GENE' and entity_2_type.upper() == 'GENE':
                pmid_list.add(pmid)

                if entity_a_species != 'HUMAN':
                    if 'Tax:' + entity_a_species in entity_1_norm:
                        entity_1_correct = True
                    elif 'Tax:' + entity_a_species in entity_2_norm:
                        entity_1_reverse = True
                else:
                    if 'Tax:' not in entity_1_norm:
                        entity_1_correct = True
                    elif 'Tax:' not in entity_2_norm:
                        entity_1_reverse = True

                if entity_b_species != 'HUMAN':
                    if 'Tax:' + entity_b_species in entity_2_norm:
                        entity_2_correct = True
                    elif 'Tax:' + entity_b_species in entity_2_norm:
                        entity_2_reverse = True
                else:
                    if 'Tax:' not in entity_2_norm:
                        entity_2_correct = True
                    elif 'Tax:' not in entity_1_norm:
                        entity_2_reverse = True



                if entity_1_correct is True and entity_2_correct is True:
                    entity_a_texts.add(entity_1_text)
                    entity_b_texts.add(entity_2_text)
                    forward_sentences[label] = pubtator_sentence

                elif entity_1_reverse is True and entity_2_reverse is True:
                    entity_b_texts.add(entity_1_text)
                    entity_a_texts.add(entity_2_text)
                    reverse_sentences[label] = pubtator_sentence

                else:
                    continue


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
        return load_gene_gene_abstract_sentences(pubtator_file, entity_a_specific, entity_b_specific)


def load_distant_kb(distant_kb_file, column_a, column_b,distant_rel_col):
    '''Loads data from knowledge base into tuples'''
    distant_interactions = set()
    reverse_distant_interactions = set()
    #reads in lines from kb file
    file = open(distant_kb_file,'rU')
    lines = file.readlines()
    file.close()
    for l in lines:
        split_line = l.split('\t')
        #column_a is entity_1 column_b is entity 2
        tuple = (split_line[column_a],split_line[column_b])
        if split_line[distant_rel_col].endswith('by') is False:
            distant_interactions.add(tuple)
        else:
            reverse_distant_interactions.add(tuple)

    #returns both forward and backward tuples for relations
    return distant_interactions,reverse_distant_interactions