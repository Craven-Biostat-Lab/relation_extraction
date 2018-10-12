# Relation Extraction from PubTator File

This program trains a model that identifies biological relations between entities using distant supervision.
The pubtator annotated files are loaded from https://zenodo.org/record/1243969#.W7_xsRNKiu4 . The HIV-1 Human Interaction
database is used for the distant supervision. https://www.ncbi.nlm.nih.gov/genome/viruses/retroviruses/hiv-1/interactions/


## Getting Started

Virtual Environment is recommended to be built using requirements.txt

### Prerequisites


```
Most important libraries
TensorFlow 1.5
NumPy 1.14
```

## Running
###DISTANT TRAINING
```
python relation_extraction_pubtator.py <MODE> <PUBTATOR_FILE> <DIRECTIONAL_DIRECTORY> <SYMMETRIC_DIRECTORY> <ENTITY_1_COLUMN> <ENTITY_2_COLUMN> <RELATION_COLUMN> <ENTITY_1> <ENTITY_2>
```

<li>MODE: TRAIN_FEED_FORWARD TRAIN_RECURRENT</li>
<li>MODEL_DIRECTORY: Path to where to save model once trained</li>
<li>PUBTATOR_FILE: Pubtator annotated file in format from https://zenodo.org/record/1243969#.W7_xsRNKiu4</li>
<li>DIRECTIONAL_DIRECTORY: Directory with files for different directional relations (regulates,degrades,etc.) for distant supervision</li>
<li>SYMMETRIC_DIRECTORY: Directory with files for different symmetric relations (binds,colocalizes,etc.) for distant supervision</li>
<li>ENTITY_1_COLUMN: Column in distant supervision files (previous 2 commands) for Entity 1</li>
<li>ENTITY_2_COLUMN: Column in distant supervision files (previous 2 commands) for Entity 2</li>
<li>RELATION_COLUMN: Column in distant supervision files (previous 2 commands) for Relation of interest</li>
<li>ENTITY_1: one of the entities you're interested in, in format SPECIES_GENE i.e. 11676_GENE for HIV-1 genes (you can put NONE for species, NONE_GENE)</li>
<li>ENTITY_2: the other entities you're interested in, in format SPECIES_GENE i.e. 9606_GENE for HUMAN genes (you can put NONE for species, NONE_GENE)</li>

###PREDICTION

```
python relation_extraction_pubtator.py PREDICT <MODEL_DIRECTORY> <PUBTATOR_FILE> <ENTITY_1> <ENTITY_2> <RECURRENT_BOOL>
```
<li>MODEL_DIRECTORY: Path of where model is trained__
<li>PUBTATOR_FILE: Pubtator annotated file for predictions you want to make in format from https://zenodo.org/record/1243969#.W7_xsRNKiu4</li>
<li>ENTITY_1: one of the entities you're interested in, in format SPECIES_GENE i.e. 11676_GENE for HIV-1 genes (you can put NONE for species, NONE_GENE)</li>
<li>ENTITY_2: the other entities you're interested in, in format SPECIES_GENE i.e. 9606_GENE for HUMAN genes (you can put NONE for species, NONE_GENE)</li>
<li>RECURRENT_BOOL: Boolean if trained model you're predicting from is recurrent or feed forward. True for recurrent. False for Feed Forward</li>

