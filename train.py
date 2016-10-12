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

#Set model parameters here
EMBEDDING_DIM = 128
VOCAB_SIZE = 10**4
MAX_SENTENCE_LENGTH = 50
NUM_CLASSES = 2

#Set training parameters here
BATCH_SIZE = 64
NUM_EPOCHS = 1

#function to save model parameters
def write_summary(output_file, train_acc, train_loss, dev_acc, dev_loss):

	with open(output_file) as outfile:

		outfile.write('EMBEDDING_DIM: ' + str(EMBEDDING_DIM))
		outfile.write('\nVOCAB_SIZE: ' + str(VOCAB_SIZE))
		outfile.write('\nMAX_SENTENCE_LENGTH: ' + str(MAX_SENTENCE_LENGTH))
		outfile.write('\nNUM_CLASSES:' + str(NUM_CLASSES))
		outfile.write('\nBATCH_SIZE: ' + str(BATCH_SIZE))
		outfile.write('\nNUM_EPOCHS: ' + str(NUM_EPOCHS))

		outfile.write('\n\nTrain Accuracy: ' + str(train_acc))
		outfile.write('\nTrain Loss: ' + str(train_loss))

		outfile.write('\n\nDev Accuracy: ' + str(dev_acc))
		outfile.write('\nDev Loss: ' + str(dev_loss))


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

		#Define training procedure
		global_step = tf.Variable(0, name='global_step', trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cbow.loss)
		train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

		#Initialize all variables
		sess.run(tf.initialize_all_variables())

		def train_step(x_batch, y_batch):

			feed_dict = {
				cbow.input_x: x_batch,
				cbow.input_y: y_batch
			}
			step,loss,accuracy = sess.run(
				[global_step, cbow.loss, cbow.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print "Train - {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
			return loss, accuracy

        #Generate batches
        x_batches, y_batches = utils.get_batches(train_data, train_labels, batch_size, num_epochs)

        #Training loop
        train_acc = None
        train_loss = None
        for i in range(len(x_batches)):
        	x_batch = x_batches[i]
        	y_batch = y_batches[i]
        	train_acc, train_loss = train_step(x_batch, y_batch)
        	current_step = tf.train.global_step(sess, global_step)

        #Report dev accuracy
        dev_feed_dict = {
        	cbow.input_x: dev_data,
        	cbow.input_y: dev_labels
        }
        dev_step, dev_loss, dev_acc = sess.run(
        	[global_step, cbow.loss, cbow.accuracy]
        	)
        print "Dev: step {}, loss {:g}, acc {:g}".format(dev_step, dev_loss, dev_acc)

        #Save model in directory with name = current UTC timestamp
        save_directory = time.time()
        outdir = os.getcwd()+save_directory+'/'
        saver = tf.train.Saver(tf.all_variables())
        if not os.path.exists(outdir):
        	write_summary(outdir+'params.txt', train_acc, train_loss, dev_acc, dev_loss)
        	path = saver.save(sess, save_directory, global_step=global_step)
        	print 'Saved model and summary to ' + path
        else:
        	#this will never happen... 
        	print 'Unable to save model'

