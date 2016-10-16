'''
Aditi Nair (asn264)
October 8 2016
'''

import os
import re
import sys
import numpy as np
from nltk.util import ngrams
from collections import Counter

PATH_TO_TRAIN = '/Users/aditinair/Desktop/NLU_DL/assignment2/data/aclImdb/train/'
PATH_TO_TEST = '/Users/aditinair/Desktop/NLU_DL/assignment2/data/aclImdb/train/'

def get_data(train=False, test=False):

	if train:
		path = PATH_TO_TRAIN
	elif train:
		path = PATH_TO_TEST
	else:
		print 'Specify train or test for get_data function'
		sys.exit()

	data = []
	labels = []

	for f in os.listdir(path+'pos/'):

		#ignore hidden files
		if not f.startswith('.'):

			with open(path+'pos/'+f) as review:

				data.append(clean_str(review.read()))

				#this is a probability distribution over class membership
				labels.append([1,0])

	
	for f in os.listdir(path+'neg/'):

		#ignore hidden files
		if not f.startswith('.'):

			with open(path+'neg/'+f) as review:

				data.append(clean_str(review.read()))

				#this is a probability distribution over class membership
				labels.append([0,1])
	
	#list of strings, list of ints
	return np.asarray(data), np.asarray(labels)


def clean_str(string):

	'''
	From Denny Britz CNN tutorial, who originally borrowed it from:
	https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	'''
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	#my own addition to get rid of weird break symbols
	string = string.replace("<br \>", " \n")

	return string.strip().lower()


def get_vocabulary(data, ngram_values=[1,2], vocab_sizes=[10**4,10**2]):

	'''
	ngrams is should be explicit list of the n-grams to include - so [1,3] will have unigrams and trigrams
	vocab_sizes should have the corresponding vocab sizes for each n-gram
	'''

	#Represent the data as a list of lists - one sub-list per review
	corpus = []
	for review in data:
		corpus.append(review.split())

	vocabulary = {'PADDING_TOKEN': 0}

	#For each n value, get the top X n-grams in the corpus
	for idx, n in enumerate(ngram_values):

		#ngrams will be a list of all the ngrams in the corpus (for current n)
		n_grams = []		
		for review in corpus:
			n_grams.extend( [' '.join(grams) for grams in ngrams(review,n)] )

		#Count on ngrams
		counts = Counter(n_grams).most_common(vocab_sizes[idx])

		#Get the maximum idx in the dictionary
		max_idx = max(vocabulary.values())
		for gram_idx,gram in enumerate(counts):
			#gram_idx starts at zero so add 1 always. gram is (token, count) so only take token.
			vocabulary[gram[0]]=(max_idx+gram_idx+1)
	
	return vocabulary


def get_truncated_bow(vocabulary, data, max_sentence_length=50, ngram_values=[1,2]):

	'''
	For each review:
	1. truncate sentence
	2. get all the required ngrams
	3. transfer them to idx
	'''

	max_num_tokens = sum([max_sentence_length-val+1 for val in ngram_values])

	bow_data = []
	for review in data:
		
		#Cut off sentence lengths 
		review_tokens = review.strip().split()[:max_sentence_length]

		#Get idx for all ngrams in current review
		curr_bow = []
		for n in ngram_values:

			for ngram in [' '.join(grams) for grams in ngrams(review_tokens,n)]:

				try:
					curr_bow.append(vocabulary[ngram])
				except KeyError:
					curr_bow.append(vocabulary['PADDING_TOKEN'])
				if len(curr_bow) == max_num_tokens:
					break
			if len(curr_bow) == max_num_tokens:
				break

		#Padding if needed
		diff = max_num_tokens - len(curr_bow)
		if diff > 0:
			curr_bow+=[vocabulary['PADDING_TOKEN']]*diff

		bow_data.append(np.asarray(curr_bow))

	return np.asarray(bow_data)


def shuffled_train_dev_split(data, labels, train_frac=0.8):

	np.random.seed(10)
	shuffled_idx = np.random.permutation(len(data))
	train_split = int(np.floor(len(data)*train_frac))

	shuffled_data = np.asarray(data)[shuffled_idx]
	shuffled_labels = np.asarray(labels)[shuffled_idx]

	return shuffled_data[:train_split], shuffled_labels[:train_split], shuffled_data[train_split:], shuffled_labels[train_split:]


def batch_iterator(data, labels, batch_size, num_epochs=1):

	data = np.copy(np.asarray(data))
	labels = np.copy(np.asarray(labels))

	data_size = len(data)
	num_partitions = data_size/int(batch_size)

	for j in range(num_epochs):

		c_idx = 0
		for i in range(num_partitions):
			
			yield np.asarray(data[c_idx:c_idx+batch_size]), np.asarray(labels[c_idx:c_idx+batch_size])
			c_idx += batch_size
		
		if num_partitions*batch_size != data_size:
			
			yield np.asarray(data[c_idx:]), np.asarray(labels[c_idx:])

		shuffled_idx = np.random.permutation(len(data))
		data = data[shuffled_idx]
		labels = labels[shuffled_idx]


def get_batch_mask(x_batch,embedding_dim,max_num_tokens):

        mask = None
        for batch in x_batch:
                c_mask = None
                for idx in batch:
                        if idx == 0:
                                if c_mask is None:
                                        c_mask = np.zeros(embedding_dim)
                                else:
                                        c_mask = np.vstack([c_mask,np.zeros(embedding_dim)])
                        else:
                                if c_mask is None:
                                        c_mask = np.ones(embedding_dim)
                                else: 
                                        c_mask = np.vstack([c_mask,np.ones(embedding_dim)])
                if mask is None:
                        mask = c_mask
                else:
                        mask = np.vstack([mask,c_mask])

        return mask.reshape((len(x_batch),max_num_tokens,embedding_dim))


def get_batch_nonzeros(x_batch,embedding_dim):

	nonzero_counts = []
	for batch in x_batch:
		#count the number of non-zero idx in the batch, then make an array of size embedding_dim 
		nonzero_counts.append(np.full(embedding_dim,np.count_nonzero(batch)))

	return np.array(nonzero_counts)


def main():

	#Little test data set for ngram tweak of get_vocabulary
	data = ['Its now pretty clear that Donald Trump has been using his presidential campaign to promote his various business ventures.',
	 'Remember when he touted his Turnberry, Scotland, golf course as a beneficiary of Great Britains exit from the European Union this summer?',
	 'But if Trump hoped his campaign would elevate the value of his brand, it looks like just the opposite is happening.']

	vocabulary = get_vocabulary(data, ngram_values=[1,2], vocab_sizes=[10,5])
	bow_trunc = get_truncated_bow(vocabulary, data, max_sentence_length=10, ngram_values=[1,2])

	print vocabulary
	print bow_trunc


if __name__ == '__main__':

	try:
		main()

	except EOFError, KeyboardInterrupt:
		sys.exit()

