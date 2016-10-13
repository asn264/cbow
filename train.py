'''
Aditi Nair
October 8 2016

Takes training data and trains a CBOW model
'''
import os
import sys
import time
import utils
import pickle
import datetime
from cbow import *
import tensorflow as tf
import numpy as np

#Set model parameters here
EMBEDDING_DIM = 128
VOCAB_SIZE = 10**4
MAX_SENTENCE_LENGTH = 50
NUM_CLASSES = 2

#Set training parameters here
BATCH_SIZE = 64
NUM_EPOCHS = 1


#Function to pickle model parameters as dictionary
def write_summary(output_file):

	with open(output_file, 'wb') as outfile:

		params = {
			'EMBEDDING_DIM': EMBEDDING_DIM,
			'VOCAB_SIZE': VOCAB_SIZE,
			'MAX_SENTENCE_LENGTH': MAX_SENTENCE_LENGTH,
			'NUM_CLASSES': NUM_CLASSES,
			'BATCH_SIZE': BATCH_SIZE,
			'NUM_EPOCHS': NUM_EPOCHS
		}

		pickle.dump(params,outfile)


#Load data
data, labels = utils.get_data(train=True)

#Get the vocabulary
vocabulary = utils.get_vocabulary(data, vocab_size=VOCAB_SIZE)

#Get a truncated bow
bow_data = utils.get_truncated_bow(vocabulary, data, max_sentence_length=MAX_SENTENCE_LENGTH)

#Train-dev split (also shuffles)
train_data, train_labels, dev_data, dev_labels = utils.shuffled_train_dev_split(bow_data, labels, train_frac=0.80)


# TRAINING
# ---------------------------------------------------------------------------------------------------------
with tf.Graph().as_default():

	sess = tf.Session()

	with sess.as_default():

		cbow = CBOW(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)

		#Define training procedure
		global_step = tf.Variable(0, name='global_step', trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cbow.loss)
		train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

		#Initialize all variables
		sess.run(tf.initialize_all_variables())

		train_accuracy = None
		train_loss = None
		def train_step(x_batch, y_batch):

			feed_dict = {
				cbow.input_x: x_batch,
				cbow.input_y: y_batch
			}
			_, train_step,train_loss,train_accuracy = sess.run(
				[train_op, global_step, cbow.loss, cbow.accuracy],
				feed_dict)
			print "Training: step {}, loss {:g}, acc {:g}".format(train_step, train_loss, train_accuracy)

        #Generate batches
        batch_iter = utils.batch_iterator(train_data, train_labels, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        for i in batch_iter:
        	x_batch = i[0]
        	y_batch = i[1]
        	train_step(x_batch, y_batch)
        	current_step = tf.train.global_step(sess, global_step)

        #Report dev accuracy
        dev_feed_dict = {
        	cbow.input_x: dev_data,
        	cbow.input_y: dev_labels
        }
        dev_step, dev_loss, dev_accuracy = sess.run(
        	[global_step, cbow.loss, cbow.accuracy],
        	dev_feed_dict)
        print "Dev: step {}, loss {:g}, acc {:g}".format(dev_step, dev_loss, dev_accuracy)

        #Save model in directory with name = current UTC timestamp
        save_directory_name = str(int(time.time()))
        outdir = os.getcwd()+'/'+save_directory_name+'/'

        #Check if the directory already exists
        if not os.path.exists(outdir):

        	#Make the directory
        	os.makedirs(outdir)
        	
        	#Write summary to text file with params, accs and losses
        	write_summary(outdir+'params.p')

        	#Pickle the vocabulary dict
        	with open(outdir+'vocabulary.p', 'wb') as vocab_pickle:
	        	pickle.dump(vocabulary, vocab_pickle)

        	#tf Saver
        	saver = tf.train.Saver(tf.all_variables())
        	path = saver.save(sess, outdir+'model.saved', global_step=global_step)
        	print 'Saved model and summary to ' + path

        #this will never happen bc directory names are all unix timestamps...
        else:
        	print 'Unable to save model'
