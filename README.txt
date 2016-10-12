Author: Aditi Nair
Date: October 5 2016

- Methods
- Explain files
- Requirements

CBOW Model:
- Read in dataset
- Extract the 10k-1 most frequent words
- Then the features are these words + UNK for the rest
- For each review get the BOW representation according to this featurization
- Have an embedding matrix E: 10K by embedding dim
- Each word in the vocabulary is associated with a row in E
- For each review, pick out the rows corresponding to its BOW
- Then average the rows
- Then pass through an MLP: another weight matrix follows by a non-linearity, then softmax for prediction. 

- Training: