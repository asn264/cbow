'''
Aditi Nair (asn264)
October 8 2016
'''

import os
import sys
import numpy as np
from collections import Counter

PATH_TO_TRAIN = '/Users/aditinair/Desktop/NLU_DL/assignment2/data/aclImdb/train/'
PATH_TO_TEST = '/Users/aditinair/Desktop/NLU_DL/assignment2/data/aclImdb/train/'

def get_data(train=False, test=False):

	data = []
	labels = []

	if train:
		for f in os.listdir(PATH_TO_TRAIN+'pos/'):

			#ignore hidden files
			if not f.startswith('.'):

				with open(PATH_TO_TRAIN+'pos/'+f) as review:

					data.append(review.read())

					#this is a probability distribution over class membership
					labels.append([1,0])

	if test:
		for f in os.listdir(PATH_TO_TEST+'neg/'):

			#ignore hidden files
			if not f.startswith('.'):

				with open(PATH_TOT_TEST+'neg/'+f) as review:

					data.append(review.read())

					#this is a probability distribution over class membership
					labels.append([0,1])

	#list of strings, list of ints
	return data, labels


def get_dataset(data, vocab_size=10000, max_sentence_length=50):

	'''
	Define the vocabulary with all the unique words in the train and dev set
	Get a BOW representation of each review
	'''
	
	#Get the top 10k-1 words in the corpus. Last word is UNK. Replace weird break symbols with newline chars.
	corpus = ' '.join(review.replace('<br />',' \n') for review in data).lower().split()
	counts = Counter(corpus).most_common(vocab_size-1)
	vocabulary = {token[0]:idx for idx,token in enumerate(counts)}

	#Add UNK to vocab, with idx = last element in dictionary list
	unk_idx = len(vocabulary)
	vocabulary['UNK'] = unk_idx

	#Clean up
	del corpus
	del counts

	#Tokenize reviews, convert to vocab idx, and throw out words not in the vocabulary. Limit max_sentence_length to 50.
	bow_data = []
	for review in data:
		
		#Tokenize reviews and limit max sentence length to 50
		truncated_bow = review.strip().split()[:max_sentence_length]

		#Convert tokens to corresponding vocab idx, if possible
		bow_by_idx = []
		for token in truncated_bow:
			try:
				bow_by_idx.append(vocabulary[token])
			except KeyError:
				bow_by_idx.append(unk_idx)
		bow_data.append(bow_by_idx)

	return bow_data, vocabulary


def shuffled_train_dev_split(data, labels, train_frac=0.8):

	np.random.seed(10)
	shuffled_idx = np.random.permutation(len(data))
	train_split = int(np.floor(len(data)*train_frac))

	shuffled_data = np.asarray(data)[shuffled_idx]
	shuffled_labels = np.asarray(labels)[shuffled_idx]

	return shuffled_data[:train_split], shuffled_labels[:train_split], shuffled_data[train_split:], shuffled_labels[train_split:]


def get_batches(data, labels, batch_size, num_epochs):

	'''
	batch_data = []
	batch_labels = []

	delta = int(np.floor(len(data)/float(batch_size)))
	data_size = len(data)

	c_idx = 0
	while len(batch_data) < data_size:
		if c_idx+delta <= data_size:
			batch_data.append(data[c_idx:c_idx+delta])
			batch_labels.append(labels[c_idx:c_idx+delta])
			c_idx += delta
		else:
			batch_data.append(data[c_idx:])
			batch_labels.append(labels[c_idx:])

	return batch_data*num_epochs, batch_labels*num_epochs
	'''

	#do this arithmetic and change batch_size
	batched_data = np.array_split(data,batch_size)
	batched_labels = np.array_split(labels,batch_size)

	return np.asarray([batched_data for i in range(num_epochs)]), np.asarray([batched_labels for i in range(num_epochs)])

def main():

	data, labels = get_data(train=True)
	vocabulary, bow_data = get_dataset(data)


if __name__ == '__main__':

	try:
		main()

	except EOFError, KeyboardInterrupt:
		sys.exit()

