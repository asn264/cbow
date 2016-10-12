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

#Set model parameters here
EMBEDDING_DIM = 128
VOCAB_SIZE = 10**4
MAX_SENTENCE_LENGTH = 50
NUM_CLASSES = 2

#Set training parameters here
BATCH_SIZE = 1
NUM_EPOCHS = 1

#Load data
data, labels = utils.get_data(train=True)

#Get the vocabulary and transform to bag of words
bow_data, vocabulary = utils.get_dataset(data, vocab_size=VOCAB_SIZE, max_sentence_length=MAX_SENTENCE_LENGTH)

#Train-dev split (also shuffles)
train_data, train_labels, dev_data, dev_labels = utils.shuffled_train_dev_split(bow_data, labels, train_frac=0.80)

# TRAINING
# ---------------------------------------------------------------------------------------------------------
with tf.Graph().as_default():

	sess = tf.Session()

	with sess.as_default():

		cbow = CBOW(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)

		#Initialize all variables
		sess.run(tf.initialize_all_variables())

		def debug_step(x_batch, y_batch):

			feed_dict = {
				cbow.input_x: x_batch,
				cbow.input_y: y_batch
			}
			input_x,input_y,E,embedded_tokens,aggregated_embedding, scores, predictions,loss,accuracy = sess.run(
				[cbow.input_x, cbow.input_y, cbow.E, cbow.embedded_tokens, cbow.aggregated_embedding, cbow.scores, cbow.predictions, cbow.loss, cbow.accuracy],
				feed_dict)
			print 'X: ', input_x.shape
			print 'Y: ', input_y.shape
			print 'E: ', E.shape
			print 'Embedded tokens: ', embedded_tokens.shape
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

