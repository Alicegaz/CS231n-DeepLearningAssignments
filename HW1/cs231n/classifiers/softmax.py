import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  for i in range(X.shape[1]):
    f_i = W.dot(X[:, i])
    f_i = f_i - np.max(f_i)
    log_C = -np.max(f_i)
    loss += -f_i[y[i]] + np.log(np.sum(np.exp(f_i)))
    for j in range(W.shape[0]):
        #d(L_i)/d(w_j) = (-log(exp(f_y_i)/sum(exp(f_ij)))) = -(f_y_i)' + (1/sum(exp(f_ij)))*sum(exp(f_ij))' = -(w_j*x_i)' + (1/sum(exp(f_i)))*(exp(x_i*w+x_i*w2+..+x_i*w_n))' = -x_i + (1/sum(exp(f_i)))*x_i*exp(x_i*w_j) = (-1 + (1/sum(exp(f_i)))*exp(x_i*w_j))*x_i
        #
        if (j == y[i]):
            dW[j, :] += (-1 + np.exp(f_i[j])/np.sum(np.exp(f_i)))*X[:, i]
        else:
            dW[j, :]+= (np.exp(f_i[j])/np.sum(np.exp(f_i)))*X[:, i]
            
  loss = loss/X.shape[1] + reg*np.sum(W*W)
  dW = dW/X.shape[1] + 2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  f = W.dot(X)
  #substract(f_i - vec(max(f_i)))
  f -= np.repeat(np.array(np.max(f, axis=0)).reshape((1, X.shape[1])), [W.shape[0]], axis=0)
  wright_scores =  f.T[range(X.shape[1]), y].T
  loss = -wright_scores + np.log(np.sum(np.exp(f), axis=0))
  loss = np.sum(loss)/X.shape[1] + reg*np.sum(W*W)
  score = (np.exp(f).T/np.array(np.sum(np.exp(f), axis = 0))[:, None])
  score[range(X.shape[1]), y] -=1
  dW = (X.dot(score)/X.shape[1]).T+2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
