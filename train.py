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
import numpy.ma as ma

#Set model parameters here
EMBEDDING_DIM = 128
MAX_SENTENCE_LENGTH = 70
NUM_CLASSES = 2
N_GRAM_VALUES = [1,2,3]
VOCAB_SIZES = [10**4,10**4,10**2]
MAX_NUM_TOKENS = sum([MAX_SENTENCE_LENGTH-val+1 for val in N_GRAM_VALUES])
L2_REG_LAMBDA = 0.5
DROPOUT_KEEP_PROB = 0.5

#Set training parameters here
BATCH_SIZE = 32
NUM_EPOCHS = 10


#Function to pickle model parameters as dictionary
def write_summary(output_file):

	with open(output_file, 'wb') as outfile:

		params = {
			'EMBEDDING_DIM': EMBEDDING_DIM,
			'VOCAB_SIZE': VOCAB_SIZES,
			'N_GRAM_VALUES': N_GRAM_VALUES,
			'MAX_SENTENCE_LENGTH': MAX_SENTENCE_LENGTH,
			'NUM_CLASSES': NUM_CLASSES,
			'BATCH_SIZE': BATCH_SIZE,
			'NUM_EPOCHS': NUM_EPOCHS,
                        'DROPOUT_KEEP_PROB': DROPOUT_KEEP_PROB,
                        'MAX_NUM_TOKENS': MAX_NUM_TOKENS
		}

		pickle.dump(params,outfile)


#Load data
data, labels = utils.get_data(train=True)

#Get the vocabulary
vocabulary = utils.get_vocabulary(data, ngram_values=N_GRAM_VALUES, vocab_sizes=VOCAB_SIZES)

#Get a truncated bow
bow_data = utils.get_truncated_bow(vocabulary, data, max_sentence_length=MAX_SENTENCE_LENGTH, ngram_values=N_GRAM_VALUES)

#Train-dev split (also shuffles)
train_data, train_labels, dev_data, dev_labels = utils.shuffled_train_dev_split(bow_data, labels, train_frac=0.80)


# TRAINING
# ---------------------------------------------------------------------------------------------------------
with tf.Graph().as_default():

	sess = tf.Session()

	with sess.as_default():

		cbow = CBOW(vocab_size=sum(VOCAB_SIZES)+1, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES, max_num_tokens=MAX_NUM_TOKENS, l2_reg_lambda=L2_REG_LAMBDA)

		#Define training procedure
		global_step = tf.Variable(0, name='global_step', trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cbow.loss)
		train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

		#Initialize all variables
		sess.run(tf.initialize_all_variables())

		def train_step(x_batch, y_batch):

                        mask = utils.get_batch_mask(x_batch,embedding_dim=EMBEDDING_DIM,max_num_tokens=MAX_NUM_TOKENS)
                        nonzero_divs = utils.get_batch_nonzeros(x_batch,embedding_dim=EMBEDDING_DIM)
			feed_dict = {
				cbow.input_x: x_batch,
                                cbow.input_x_mask: mask,
                                cbow.x_divs: nonzero_divs,
				cbow.input_y: y_batch,
                                cbow.dropout_keep_prob: DROPOUT_KEEP_PROB
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


        #Report dev accuracy - use iterator because otherwise masking is very slow on the full batch
        dev_iter = utils.batch_iterator(dev_data, dev_labels, batch_size=BATCH_SIZE, num_epochs=1)
        all_dev_predictions = []
        dev_loss_sum = 0.0
        l2_loss = None
        counter = 1

        for i in dev_iter:

                print 'Batch ' + str(counter) + ' for dev_iter'
                counter += 1

                x_batch = i[0]
                y_batch = i[1]
                mask = utils.get_batch_mask(x_batch,embedding_dim=EMBEDDING_DIM,max_num_tokens=MAX_NUM_TOKENS)
                nonzero_divs = utils.get_batch_nonzeros(x_batch,embedding_dim=EMBEDDING_DIM)
                
                dev_feed_dict = {
                        cbow.input_x: x_batch,
                        cbow.input_x_mask: mask,
                        cbow.x_divs: nonzero_divs,
                        cbow.input_y: y_batch,
                        cbow.dropout_keep_prob: 1.0
                }

                dev_loss, l2_loss, batch_preds = sess.run(
        	       [cbow.loss, cbow.l2_loss, cbow.predictions],
        	       dev_feed_dict)

                all_dev_predictions = np.concatenate([all_dev_predictions, batch_preds])
                
                #dev_loss is reported by the model as (mean loss on batch size + lambda*l2_loss)
                dev_loss_sum += (dev_loss-L2_REG_LAMBDA*l2_loss)*len(x_batch)
                
                if l2_loss is None:
                        l2_loss = L2_REG_LAMBDA*l2_loss

        #Finally report Dev results
        dev_label_values = np.argmax(dev_labels,axis=1)
        correct_dev_predictions = float(sum(ma.masked_equal(all_dev_predictions,dev_label_values).mask))
        print 'Dev Accuracy: ' + str( correct_dev_predictions/len(dev_label_values) )
        print 'Dev Loss: ' + str( (dev_loss_sum/len(dev_label_values)) + l2_loss )

        #print "Dev: step {}, loss {:g}, acc {:g}".format(dev_step, dev_loss, dev_accuracy)

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
