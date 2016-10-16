'''
Aditi Nair (asn264)
October 8 2016
'''

import tensorflow as tf
import sys

class CBOW(object):

	def __init__(self, vocab_size, embedding_dim=128, num_classes=2, max_num_tokens=50, l2_reg_lambda=0):

		#shape is [None,None] - meaning we can feed in any number of samples and each sample can have arbitrary length
		self.input_x = tf.placeholder(tf.int32, shape=[None, max_num_tokens], name='input_x')
		self.input_x_mask = tf.placeholder(tf.float32, shape=[None, max_num_tokens,embedding_dim], name='x_mask')

		#shape is [None,num_classes] - meaning we can feed in any number of samples 
		#but each sample must be labelled with a probability distribution over all classes - so floats!
		self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')

		#Dropout Regularization
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		#L2 Regularization
		self.l2_loss = tf.constant(0.0)

		#initialize the embedding matrix with random values
		self.E = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='E')

		#this should have dimension (# samples/None, max_num_tokens, embedding size)
		self.embedded_tokens = tf.nn.embedding_lookup(self.E, self.input_x)
		self.masked_tokens = tf.mul(self.embedded_tokens,self.input_x_mask,name='masked_tokens')

		#aggregate the embedding rows for each sample in the input
		self.aggregated_embedding = tf.reduce_sum(self.masked_tokens,reduction_indices=1)

		#do the mean by dividing by the number of non-zero tokens in each review
		self.x_divs = tf.placeholder(tf.float32,shape=[None,embedding_dim],name='x_divs')
		#self.aggregated_embedding = tf.truediv(self.aggregated_embedding,self.x_divs)

		#dropout
		self.aggregated_embedding_dropout = tf.nn.dropout(self.aggregated_embedding,self.dropout_keep_prob)

		#MLP
		W = tf.Variable(tf.truncated_normal([embedding_dim,num_classes]), name='W')
		b = tf.Variable(tf.constant(0.1), num_classes, name='b')

		#Wx + b for each input x gives the scores, then do nonlinearity
		self.scores = tf.nn.sigmoid(tf.matmul(self.aggregated_embedding_dropout,W)+b)

		#Prediction by argmax
		#Don't need to do softmax here bc it won't change the relative scale and therefore won't change prediction
		self.predictions = tf.argmax(self.scores, 1, name='predictions')

		#Compute loss - cross entropy of softmax on distributions + l2 loss
		self.l2_loss += tf.nn.l2_loss(W)
		self.l2_loss += tf.nn.l2_loss(b)
		losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
		self.loss = tf.reduce_mean(losses) + l2_reg_lambda*self.l2_loss

		#Compute accuracy
		correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
   		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
