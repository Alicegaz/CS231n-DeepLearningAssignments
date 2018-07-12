import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0] #number of rows
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j] += X[:,i]
        dW[y[i]] += - X[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
    
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  train_num = X.shape[1]
  dW = np.zeros(W.shape).T # initialize the gradient as zero
  scores = W.dot(X) 
  # compute the Loss function
  margin = (scores.T - scores.T[range(X.shape[1]), y].reshape(scores.shape[1], 1)).T + 1
  #replace negative margins with 0
  margin = (margin.clip(min=0)).T
  #margins of the correct class turns to be delta, replace by 0
  margin[range(X.shape[1]), y] = 0
  #compute loss
  loss =round((np.sum(np.sum(margin.T, axis=1))/train_num+0.5*reg*np.sum(W*W)), 5)
  #replace > 0 values by one, to compute bradient in the next step
  margin = ((margin.T>0).astype(int)).T
  #replace the margin of each training sample by the count of uncorrect classes
  margin[range(X.shape[1]), y] = (-np.sum(margin, axis=1)).T
  dW = margin.T.dot(X.T)/train_num+reg*W
  return loss, dW
