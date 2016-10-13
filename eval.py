import sys
import utils
import pickle
from cbow import *
import numpy as np
import numpy.ma as ma

SAVE_DIRECTORY_PATH = '/Users/aditinair/Desktop/NLU_DL/assignment2/cbow/1476331961/'

#Evaluates combined training/dev set if false
EVAL_TEST=True

#Load up vocabulary
with open(SAVE_DIRECTORY_PATH+'vocabulary.p') as vocab_path:
	vocabulary = pickle.load(vocab_path)
with open(SAVE_DIRECTORY_PATH+'params.p') as params_path:
	params = pickle.load(params_path)

if EVAL_TEST=='test':
	
	#Load data
	data, labels = utils.get_data(test=True)

	#Get truncated BoW
	bow_data = utils.get_truncated_bow(vocabulary, data, max_sentence_length=params['MAX_SENTENCE_LENGTH'])

else:

	#Load data
	data, labels = utils.get_data(train=True)

	#Get a truncated BoW
	bow_data = utils.get_truncated_bow(vocabulary, data, max_sentence_length=params['MAX_SENTENCE_LENGTH'])

#We actually want *labels* now, not probability distributions on labels
labels = np.argmax(labels,axis=1)

# Evaluation
# ============================================================================================================
checkpoint_file = tf.train.latest_checkpoint(SAVE_DIRECTORY_PATH)
graph = tf.Graph()
with graph.as_default():

	sess = tf.Session()

	with sess.as_default():

		#Load the save metagraph and restore variables
		saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))

		saver.restore(sess, checkpoint_file)

		#Get placeholders from graph
		input_x = graph.get_operation_by_name('input_x').outputs[0]

		#Tensors to evaluate 
		predictions = graph.get_operation_by_name('predictions').outputs[0]

		all_predictions = []

		batch_iter = utils.batch_iterator(bow_data, labels, params['BATCH_SIZE'], num_epochs=1)
		for i in batch_iter:
			batch_preds = sess.run(predictions, {input_x: i[0]})
			try:
				all_predictions = np.concatenate([all_predictions, batch_preds])
			except ValueError:
				print all_predictions
				print batch_preds
				sys.exit()

correct_predictions = float(sum(ma.masked_equal(all_predictions,labels).mask))

if EVAL_TEST:
	print '---Test Results---'
else:
	print '---Train Reults---'

num_samples = len(labels)
print 'Num Samples: ', str(num_samples)
print 'Accuracy: ', str( correct_predictions/num_samples )


