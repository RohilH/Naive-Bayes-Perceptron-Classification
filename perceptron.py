import numpy as np
import matplotlib.pyplot as plt

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model.

		This function will initialize a feature_dim weight vector,
		for each class.

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS]
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		"""

		self.w = np.zeros((feature_dim+1,num_class))

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		bias = 1
		print(len(self.w), len(self.w[0]))
		dotProds = [ [] for i in range(10) ]
		idx = -1
		for image in train_set:
			idx += 1
			for classType in range(10):
				dotProds[classType] = np.dot(self.w[0:784, classType], image)
			predClass = np.argmax(dotProds)
			# print(predClass, train_label[idx])
			if predClass != train_label[idx]:
				self.w[0:784, train_label[idx]] += image*bias
				self.w[784, train_label[idx]] += bias
			# else:
				self.w[0:784, predClass] -= image*bias
				self.w[784, predClass] -= bias
		# YOUR CODE HERE
		pass

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset.
			The accuracy is computed as the average of correctness
			by comparing between predicted label and true label.

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""

		idx = -1
		dotProds = [ [] for i in range(10) ]
		accuracy = 0
		pred_label = np.zeros((len(test_set)))
		low = [float('inf') for i in range(10) ]
		high = [float('-inf') for i in range(10) ]
		lowIs = [0 for i in range(10) ]
		highIs = [0 for i in range(10) ]

		for image in test_set:
			idx += 1
			for classType in range(10):
				dotProds[classType] = np.dot(self.w[0:784, classType], image)
			predClass = np.argmax(dotProds)
			if predClass == test_label[idx]:
				accuracy += 1
			pred_label[idx] = predClass
			if dotProds[test_label[idx]] < low[test_label[idx]]:
				low[test_label[idx]] = dotProds[test_label[idx]]
				lowIs[test_label[idx]] = idx
			if dotProds[test_label[idx]] > high[test_label[idx]]:
				high[test_label[idx]] = dotProds[test_label[idx]]
				highIs[test_label[idx]] = idx
		print("Low indices for 10 classes: ", lowIs)
		print("High indices for 10 classes: ", highIs)

		# YOUR CODE HERE
		accuracy = accuracy/len(test_label)
		print(accuracy)
		pass

		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters
		"""

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters
		"""

		self.w = np.load(weight_file)
