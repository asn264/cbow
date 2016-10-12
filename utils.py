'''
Aditi Nair (asn264)
October 8 2016
'''

import os
import re
import sys
import numpy as np
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

				data.append(review.read())

				#this is a probability distribution over class membership
				labels.append([1,0])

	'''
	for f in os.listdir(path+'neg/'):

		#ignore hidden files
		if not f.startswith('.'):

			with open(path+'neg/'+f) as review:

				data.append(review.read())

				#this is a probability distribution over class membership
				labels.append([0,1])
	'''
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


def get_dataset(data, vocab_size=10000, max_sentence_length=50):

	'''
	Define the vocabulary with all the unique words in the train and dev set
	Get a BOW representation of each review
	'''
	
	#Get the top 10k-1 words in the corpus. Last word is UNK. Replace weird break symbols with newline chars.
	corpus = ' '.join(clean_str(review) for review in data).split()
	counts = Counter(corpus).most_common(vocab_size-1)
	vocabulary = {token[0]:(idx+1) for idx,token in enumerate(counts)}

	#Add PADDING_TOKEN to vocab, with idx = last element in dictionary list
	padding_idx = 0
	vocabulary['PADDING_TOKEN'] = padding_idx

	#Clean up
	del corpus
	del counts

	#Tokenize reviews, convert to vocab idx, and throw out words not in the vocabulary. Limit max_sentence_length to 50.
	bow_data = []
	for review in data:
		
		#Tokenize reviews
		truncated_bow = review.strip().split()

		#Convert tokens to corresponding vocab idx, if possible
		bow_by_idx = []
		for token in truncated_bow:
			try:
				bow_by_idx.append(vocabulary[token])
			except KeyError:
				#Don't include unknowns bc Zipfian distribution
				pass
			#Grab at most fifty known words form the sentence
			if len(bow_by_idx) == max_sentence_length:
				break

		#padding if needed
		diff = max_sentence_length - len(bow_by_idx)
		if diff > 0:
			bow_by_idx += [padding_idx]*diff
		bow_data.append(np.asarray(bow_by_idx))

	return np.asarray(bow_data), vocabulary


def shuffled_train_dev_split(data, labels, train_frac=0.8):

	np.random.seed(10)
	shuffled_idx = np.random.permutation(len(data))
	train_split = int(np.floor(len(data)*train_frac))

	shuffled_data = np.asarray(data)[shuffled_idx]
	shuffled_labels = np.asarray(labels)[shuffled_idx]

	return shuffled_data[:train_split], shuffled_labels[:train_split], shuffled_data[train_split:], shuffled_labels[train_split:]


def batch_iterator(data, labels, batch_size, num_epochs=1):

	data = np.asarray(data)
	labels = np.asarray(labels)

	data_size = len(data)
	num_partitions = data_size/int(batch_size)

	for j in range(num_epochs):

		c_idx = 0
		for i in range(num_partitions):
			
			yield np.asarray(data[c_idx:c_idx+batch_size]), np.asarray(labels[c_idx:c_idx+batch_size])
			c_idx += batch_size
		
		if num_partitions*batch_size != data_size:
			
			yield np.asarray(data[c_idx:]), np.asarray(labels[c_idx:])


def main():

	'''
	data, labels = get_data(train=True)
	bow_data, vocabulary = get_dataset(data)
	train_data, train_labels, dev_data, dev_labels = shuffled_train_dev_split(bow_data, labels)
	batch_iter = batch_iterator(train_data, train_labels, batch_size=64, num_epochs=1)
	for i in batch_iter, 
		print i
	'''

	data = np.arange(11)
	labels = data+1
	batch_iter = batch_iterator(data, labels, batch_size=3, num_epochs=2)
	for i in batch_iter:
		print i
	print data
	print labels

if __name__ == '__main__':

	try:
		main()

	except EOFError, KeyboardInterrupt:
		sys.exit()

