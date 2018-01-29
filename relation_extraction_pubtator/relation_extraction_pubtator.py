import sys
import os


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

    else:
        print("usage error")


if __name__ == "__main__":
    main()