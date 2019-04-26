import numpy as np
import matplotlib.pyplot as plt
class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model.

		This function will initialize prior and likelihood, where
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		    num_value(int): number of possible values for each pixel
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset.
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		k = 1
		idx = -1
		for i in train_label:
			self.prior[i] += 1

		self.prior = [x/50000 for x in self.prior]
		# print("prior:")
		# print(self.prior, len(self.prior))

		for image in train_set:
			idx += 1
			for i in range(len(image)):
				self.likelihood[i][image[i]][train_label[idx]] += 1
		for pixel in range(len(self.likelihood)):
			for freq in range(len(self.likelihood[0])):
				for classType in range(len(self.likelihood[0][0])):
					if self.likelihood[pixel][freq][classType] == 0:
						self.likelihood[pixel][freq][classType] += k
						self.likelihood[pixel][freq][classType] = self.likelihood[pixel][freq][classType]/(self.prior[classType]*50000 + 256*k)
					else:
						self.likelihood[pixel][freq][classType] = self.likelihood[pixel][freq][classType]/(self.prior[classType]*50000)

		# print("Likelihood: ")
		# print(self.likelihood[300][100])

		# YOUR CODE HERE
		pass

	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.
			The accuracy is computed as the average of correctness
			by comparing between predicted label and true label.

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""

		accuracy = 0
		pred_label = np.zeros((len(test_set)))
		idx = -1
		counter = 0
		low = [float('inf') for i in range(10) ]
		high = [float('-inf') for i in range(10) ]
		lowIs = [0 for i in range(10) ]
		highIs = [0 for i in range(10) ]
		# YOUR CODE HERE
		for image in test_set:
			probs = np.zeros(self.num_class)
			idx += 1
			for i in range(len(image)):
				for j in range(self.num_class):
					probs[j] += np.log(self.likelihood[i][image[i]][j])
			pred_label[idx] = np.argmax(probs)
			if test_label[idx] == pred_label[idx]:
				accuracy += 1
			if probs[test_label[idx]] < low[test_label[idx]]:
				low[test_label[idx]] = probs[test_label[idx]]
				lowIs[test_label[idx]] = idx
			if probs[test_label[idx]] > high[test_label[idx]]:
				high[test_label[idx]] = probs[test_label[idx]]
				highIs[test_label[idx]] = idx
		accuracy = accuracy/len(test_label)
		print(accuracy)
		# print("Low indices for 10 classes: ", lowIs)
		# print("High indices for 10 classes: ", highIs)

		pass


		return accuracy, pred_label


	def save_model(self, prior, likelihood):
		""" Save the trained model parameters
		"""

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters
		"""

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
	    """
	    Get the feature likelihoods for high intensity pixels for each of the classes,
	        by sum the probabilities of the top 128 intensities at each pixel location,
	        sum k<-128:255 P(F_i = k | c).
	        This helps generate visualization of trained likelihood images.

	    Args:
	        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	            (# of features/pixels per image, # of possible values per pixel, # of class)
	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	            (# of features/pixels per image, # of class)
	    """
	    # YOUR CODE HERE

	    feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))
	    for pixel in range(len(likelihood)):
		    for intensity in range(128):
		        for classType in range(len(likelihood[0][0])):
		            feature_likelihoods[pixel][classType] += self.likelihood[pixel][intensity + 128][classType]
	    return feature_likelihoods
