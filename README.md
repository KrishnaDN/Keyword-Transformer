# Keyword-Transformer
Implementation of the paper "Keyword Transformer: A Self-Attention Model for Keyword Spotting".

Our implementation obtains 96.2% accuracy on the test data.


## Kaldi Data preperation
In order to extract the features from kaldi, we need to create few files for kaldi (standerd procedure).
Use the script dataset.py to create the files and change the relevent paths (line 74). This code create a folders called data/train, data/valid and data/test


## Feature Extraction
Run the script train.sh to extract the feature. This script will extract and save features. It also creates a folder called manifest with files train, test and dev in it
```
sh train.sh
```

## Training
Finally you can run trainer.py to start the training. This code will train and save the checkpoints.
```
python trainer.py
```

## Testing
For testing, you can use testing.py with relevent file paths


