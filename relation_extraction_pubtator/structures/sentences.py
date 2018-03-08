class Sentence(object):
    def __init__(self, pmid, sentence_no, start_entity_text, start_entity_loc, end_entity_text, end_entity_loc,
                 start_entity_raw_string, end_entity_raw_string, start_entity_full_norm, end_entity_full_norm, start_entity_type, end_entity_type,
                 start_entity_id, end_entity_id, start_entity_species, end_entity_species, dep_parse, sentence):
        '''Constructor for Sentence Object'''
        self.pmid = pmid
        self.sentence_no = sentence_no
        self.start_entity_text = start_entity_text
        self.start_entity_loc = start_entity_loc
        self.end_entity_text = end_entity_text
        self.end_entity_loc = end_entity_loc
        self.start_entity_raw_string = start_entity_raw_string
        self.end_entity_raw_string = end_entity_raw_string
        self.start_entity_full_norm = start_entity_full_norm
        self.end_entity_full_norm = end_entity_full_norm
        self.start_entity_type = start_entity_type
        self.end_entity_type = end_entity_type
        self.start_entity_id = start_entity_id
        self.end_entity_id = end_entity_id
        self.start_entity_species = start_entity_species
        self.end_entity_species = end_entity_species
        self.dep_parse = dep_parse
        self.sentence_words = sentence



        #self.build_dependency_features()

    def set_start_entity_species(self, species):
        self.start_entity_species = species

    def set_end_entity_species(self, species):
        self.end_entity_species = species

    def set_start_entity_id(self, splitter, col):
        self.start_entity_id = self.start_entity_full_norm.split(splitter)[col]

    def set_end_entity_id(self, splitter, col):
        self.end_entity_id = self.end_entity_full_norm.split(splitter)[col]
