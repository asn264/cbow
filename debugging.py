'''
Aditi Nair
October 8 2016

Takes training data and trains a CBOW model
'''
import os
import sys
import datetime
import utils
from cbow import *
import tensorflow as tf
import numpy as np
import numpy.ma as ma

#Set model parameters here
EMBEDDING_DIM = 128
MAX_SENTENCE_LENGTH = 50
NUM_CLASSES = 2
N_GRAM_VALUES = [1,2]
VOCAB_SIZES = [10**4,10**4]
MAX_NUM_TOKENS = sum([MAX_SENTENCE_LENGTH-val+1 for val in N_GRAM_VALUES])
L2_REG_LAMBDA = 0.5

#Set training parameters here
BATCH_SIZE = 32
NUM_EPOCHS = 10

#Load data
data, labels = utils.get_data(train=True)

#Get the vocabulary
vocabulary = utils.get_vocabulary(data, ngram_values=N_GRAM_VALUES, vocab_sizes=VOCAB_SIZES)

#Get a truncated bow
bow_data = utils.get_truncated_bow(vocabulary, data, max_sentence_length=MAX_SENTENCE_LENGTH, ngram_values=N_GRAM_VALUES)

#Train-dev split (also shuffles)
train_data, train_labels, dev_data, dev_labels = utils.shuffled_train_dev_split(bow_data, labels, train_frac=0.80)

# DEBUGGING
# ---------------------------------------------------------------------------------------------------------
with tf.Graph().as_default():

	sess = tf.Session()

	with sess.as_default():

		cbow = CBOW(vocab_size=sum(VOCAB_SIZES)+1, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES, max_num_tokens=MAX_NUM_TOKENS, l2_reg_lambda=L2_REG_LAMBDA)

		#Initialize all variables
		sess.run(tf.initialize_all_variables())


		def get_batch_mask(x_batch):

			mask = None
			for batch in x_batch:
				c_mask = None
				for idx in batch:
					if idx == 0:
						if c_mask is None:
							c_mask = np.zeros(EMBEDDING_DIM)
						else:
							c_mask = np.vstack([c_mask,np.zeros(EMBEDDING_DIM)])
					else:
						if c_mask is None:
							c_mask = np.ones(EMBEDDING_DIM)
						else: 
							c_mask = np.vstack([c_mask,np.ones(EMBEDDING_DIM)])
				if mask is None:
					mask = c_mask
				else:
					mask = np.vstack([mask,c_mask])

			return mask.reshape((len(x_batch),MAX_NUM_TOKENS,EMBEDDING_DIM))


		def debug_step(x_batch, y_batch):

			mask = get_batch_mask(x_batch)

			feed_dict = {
				cbow.input_x: x_batch,
				cbow.input_x_mask: mask,
				cbow.input_y: y_batch
			}
			input_x,input_y,E,embedded_tokens,aggregated_embedding,scores, predictions,loss,accuracy,masked_tokens  = sess.run(
				[cbow.input_x, cbow.input_y, cbow.E, cbow.embedded_tokens, cbow.aggregated_embedding, cbow.scores, 
				cbow.predictions, cbow.loss, cbow.accuracy, cbow.masked_tokens],
				feed_dict)

			print 'X: ', input_x.shape
			print 'Y: ', input_y.shape
			print 'E: ', E.shape
			print 'Embedded tokens: ', embedded_tokens.shape
			print 'Mask: ', masked_tokens.shape
			print 'Aggregated embedding: ', aggregated_embedding.shape
			print 'Scores: ', scores.shape
			print 'Predictions: ', predictions
			print 'Loss: ', loss
			print 'Accuracy: ', accuracy

		batch_iter = utils.batch_iterator(train_data, train_labels, batch_size=3, num_epochs=1)
		for i in batch_iter:
			x_batch = i[0]
			y_batch = i[1]
			debug_step(x_batch, y_batch)
			break

