'''
Aditi Nair (asn264)
October 8 2016
'''

import tensorflow as tf
import sys

class CBOW(object):

	def __init__(self, vocab_size=10**4, embedding_dim=128, num_classes=2, batch_size=1):

		#shape is [None,None] - meaning we can feed in any number of samples and each sample can have arbitrary length
		self.input_x = tf.placeholder(tf.int32, shape=[None, batch_size], name='input_x')

		#shape is [None,num_classes] - meaning we can feed in any number of samples 
		#but each sample must be labelled with a probability distribution over all classes - so floats!
		self.input_y = tf.placeholder(tf.float32, shape=[num_classes, batch_size], name='input_y')

		#initialize the embedding matrix with random values
		self.E = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='E')

		#this should have dimension (# samples/None, sentence length, embedding size)
		self.embedded_tokens = tf.nn.embedding_lookup(self.E, self.input_x)

		#aggregate the embedding rows for each sample in the input
		#self.aggregated_embedding = tf.expand_dims(tf.reduce_mean(self.embedded_tokens,reduction_indices=0),-1)
		self.aggregated_embedding = tf.reduce_mean(self.embedded_tokens,reduction_indices=0)

		#MLP
		W = tf.Variable(tf.truncated_normal([embedding_dim,num_classes]), name='W')
		b = tf.Variable(tf.constant(0.1), num_classes, name='b')

		#Wx + b for each input x gives the scores, then do nonlinearity
		#self.scores = tf.nn.relu( tf.matmul(self.aggregated_embedding, W) + b )
		self.scores = tf.matmul(self.aggregated_embedding,W)

		#Prediction by argmax
		#Don't need to do softmax here bc it won't change the relative scale and therefore won't change prediction
		self.predictions = tf.argmax(self.scores, 1, name='predictions')

		#Compute loss - cross entropy of softmax on distributions
		self.loss = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)

		'''
		#Compute accuracy
		correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
   		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
   		'''