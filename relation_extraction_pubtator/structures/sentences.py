class Sentence(object):
    def __init__(self,pmid,sentence_no,entity_1_text,entity_1_loc,entity_2_text,entity_2_loc,
                                          entity_1_formal,entity_2_formal,entity_1_norm,entity_2_norm,entity_1_type, entity_2_type,
                 entity_1_simple_norm, entity_2_simple_norm, entity_1_species, entity_2_species,dep_parse, sentence):
        '''Constructor for Sentence Object'''
        self.pmid = pmid
        self.sentence_no = sentence_no
        self.entity_1_text = entity_1_text
        self.entity_1_loc = entity_1_loc
        self.entity_2_text = entity_2_text
        self.entity_2_loc = entity_2_loc
        self.entity_1_formal = entity_1_formal
        self.entity_2_formal = entity_2_formal
        self.entity_1_norm = entity_1_norm
        self.entity_2_norm = entity_2_norm
        self.entity_1_type = entity_1_type
        self.entity_2_type = entity_2_type
        self.entity_1_simple_norm = entity_1_simple_norm
        self.entity_2_simple_norm = entity_2_simple_norm
        self.entity_1_species = entity_1_species
        self.entity_2_species = entity_2_species
        self.dep_parse = dep_parse
        self.sentence_words = sentence



        #self.build_dependency_features()

    def set_entity_1_species(self,species):
        self.entity_1_species = species

    def set_entity_2_species(self,species):
        self.entity_2_species = species

    def set_entity_1_simple_norm(self,splitter,col):
        self.entity_1_simple_norm = self.entity_1_norm.split(splitter)[col]

    def set_entity_2_simple_norm(self,splitter,col):
        self.entity_2_simple_norm = self.entity_2_norm.split(splitter)[col]
