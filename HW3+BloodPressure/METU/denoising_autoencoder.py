import numpy as np
import matplotlib.pyplot as plt

identity = lambda x: x

class DenoisingAutoencoder(object):
    """
    Denoising autoencoder.
    corrupts the input and tries to reconstruct it
    """
    def sigmoid(self, x):
      #
      # TODO: implement sigmoid
      #
      return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
      #
      # TODO: implement sigmoid derivative
      #
      return self.ac_func(x)*(1-self.ac_func(x))

    def tanh(self, x):
      return (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))

    def tanh_deriv(self, x):
      return (1-self.tanh(x)**2)

    def ac_func(self, x, function_name = 'SIGMOID'):
        # Implement your activation function here
        fname_upper = function_name.upper()
        if fname_upper =='SIGMOID':
            return self.sigmoid(x)
        elif fname_upper == 'TANH':
          return self.tanh(x)
        else:
          raise fname_upper + " Not implemented Yet"

    def MSE(self, X, y):
      return 0.5*np.sum((X-y)**2)/X.shape[0]

    def ac_func_deriv(self, x, function_name = 'SIGMOID'):
        # Implement the derivative of your activation function here
        fname_upper = function_name.upper()
        if fname_upper == 'SIGMOID':
            return self.sigmoid_deriv(x)
        elif fname_upper=='TANH':
            return self.tanh_deriv(x)
        else:
          raise fname_upper + " Not implemented Yet"

    def __init__(self, layer_units, weights=None):
        self.weights = weights
        self.layer_units = layer_units

    def init_weights(self, seed=0):
        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        # Note layer_units[2] = layer_units[0]
        layer_units = self.layer_units
        n_layers = len(layer_units)
        assert n_layers == 3

        np.random.seed(seed)

        # Initialize parameters randomly based on layer sizes
        r  = np.sqrt(6) / np.sqrt(layer_units[1] + layer_units[0])
        # We'll choose weights uniformly from the interval [-r, r)
        weights = [{} for i in range(n_layers - 1)]
        weights[0]['W'] = np.random.random((layer_units[0], layer_units[1])) * 2.0 * r - r
        weights[1]['W'] = np.random.random((layer_units[1], layer_units[2])) * 2.0 * r - r
        weights[0]['b'] = np.zeros(layer_units[1])
        weights[1]['b'] = np.zeros(layer_units[2])

        self.weights = weights

        return self.weights

    def predict(self, X_noisy, reg=3e-3, activation_function='SIGMOID'):
        weights = self.weights

              # Weight parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']
        W1 = weights[1]['W']
        b1 = weights[1]['b']

        # TODO: Implement forward pass here. It should be the same forward pass that you implemented in the loss function
        y1 = np.dot(X_noisy, W0)+b0
        h1 = self.ac_func(y1, function_name=activation_function)
        y2 = np.dot(h1, W1)+b1
        scores = self.ac_func(y2, function_name=activation_function)
        return scores

    def loss(self, X_noisy, X, reg=3e-3, activation_function='SIGMOID'):
        weights = self.weights

        # Weighting parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']
        W1 = weights[1]['W']
        b1 = weights[1]['b']

        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the  scores for the input. 	    #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, N).                                                             #
        #############################################################################
        #y1 = np.dot(X_noisy, W0)+b0
        #h1 = self.ac_func(y1, function_name='TANH')
        #y2 = self.ac_func(np.dot(h1, W1)+b1)
        y1 = np.dot(X_noisy, W0)+b0
        h1 = self.ac_func(y1)
        y2 = np.dot(h1, W1)+b1
        scores = self.ac_func(y2, function_name=activation_function)
 #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO: Compute the loss. This should include 				    #
        #             (i) the data loss (square error loss),			    #
        #             (ii) L2 regularization for W1 and W2, and    		    #
        # Store the result in the variable loss, which should be a scalar.          #
        # (Don't forget to investigate the effect of L2 loss)                       #
        #############################################################################
        loss = self.MSE(scores, X)+0.5*reg*(np.sum(W0**2)+np.sum(W1**2))/X.shape[0]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        dL = scores-X
        dscores = dL*self.ac_func_deriv(y2, function_name=activation_function)
        dy2 = np.dot(dscores, W1.T)
        dh1 = dy2*self.ac_func_deriv(h1)
        dW0 = np.dot(X.T, dh1)
        dW1 = np.dot(h1.T, dscores)
        #dscores = (scores-X)
        #print("has to be", y2.shape, "params", dscores.shape, self.ac_func_deriv(y2).shape)
        #dy2 = np.dot(dscores, self.ac_func_deriv(y2).T)
        #print(dy2.shape)
        #print("has to be", h1.shape, "params", dy2.shape, W1.shape)
        #dh1 = np.dot(dy2, W1.T)
        #print("has to be", W1.shape, "params", dy2.shape, h1.shape)
        #dW1 = np.dot(dy2, h1)
        #print("has to be", y1.shape, "params", dh1.shape, (1-h1**2).shape)
        #dy1 = np.dot(dh1, (1-h1**2))
        #print("has to be", W0.shape, "params", dy1.shape, X_noisy.shape)
        #dW0 = np.dot(dy1, X_noisy)
        #print(dscores.shape, W1.shape)
        #dh1 = np.dot(dscores, W1.T)
        #dW1 = np.dot(dscores.T, h1).T
        #print("has to be", y1.shape, "params", dh1.shape, (h1*(1-h1)).shape)
        #dy1 = np.dot(dh1.T, (h1*(1-h1)))
        #print("has to be", y1.shape, "result ", dy1.shape, "params", dh1.shape, (h1*(1-h1)).shape)
        #print("has to be", W0.shape, "params", dy1.shape, X_noisy.shape)
        #dW0 = np.dot(dy1, X_noisy)
        #print("has to be", W0.shape, "result ", dW0.shape, "params", dy1.shape, X_noisy.shape)
        db0 = np.sum(dh1, axis=0)
        db1 = np.sum(dscores, axis=0)
        dW0 += reg*W0
        dW1 += reg*W1
        
        grads['W0'] = dW0
        grads['W1'] = dW1
        grads['b0'] = db0
        grads['b1'] = db1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train_with_SGD(self, X, noise=identity,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=3e-3, num_iters=100,
            batchsize=128, momentum='classic', mu=0.9, verbose=False,
            activation_function='sigmoid'):

            num_train = X.shape[0]

            loss_history = []

            layer_units = self.layer_units
            n_layers = len(layer_units)
            velocity = [{} for i in range(n_layers - 1)]
            velocity[0]['W'] = np.zeros((layer_units[0], layer_units[1]))
            velocity[1]['W'] = np.zeros((layer_units[1], layer_units[2]))
            velocity[0]['b'] = np.zeros(layer_units[1])
            velocity[1]['b'] = np.zeros(layer_units[2])

            for it in range(num_iters):

                  batch_indicies = np.random.choice(num_train, batchsize, replace = False)
                  X_batch = X[batch_indicies]

                  # Compute loss and gradients
                  noisy_X_batch = noise(X_batch)
                  loss, grads = self.loss(noisy_X_batch, X_batch, reg, activation_function=activation_function)
                  loss_history.append(loss)
                  self.weights[0]['W'] += -learning_rate*grads['W0']
                  self.weights[1]['W'] += -learning_rate*grads['W1']
                  self.weights[0]['b'] += -learning_rate*grads['b0']
                  self.weights[1]['b'] += -learning_rate*grads['b1']
                  #########################################################################
                  # TODO: Use the gradients in the grads dictionary to update the         #
                  # parameters of the network (stored in the dictionary self.params)      #
                  # using gradient descent.                                               #
                  #########################################################################


                  # You can start and test your implementation without momentum. After
                  # making sure that it works, you can add momentum


                  #########################################################################
                  #                             END OF YOUR CODE                          #
                  #########################################################################

                  if verbose and it % 10 == 0:
                        print('SGD: iteration %d / %d: loss %f' % (it, num_iters, loss))

                  # Every 5 iterations.
                  #if it % 5 == 0 and it<=150:
                        # Decay learning rate
                  #      learning_rate *= learning_rate_decay
                  #if it>150:
                  #  learning_rate *= 0.8
                  #if it>200 and it % 5 == 0:
                  #      learning_rate *= learning_rate_decay
                  if it % 5 == 0:
                        learning_rate *= learning_rate_decay

            return { 'loss_history': loss_history, }
