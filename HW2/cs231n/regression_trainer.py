import numpy as np
from cs231n.layers import *
import math

class RegressionTrainer(object):
    """ The trainer class performs SGD with momentum on a cost function """

    def __init__(self):
        self.step_cache = {}  # for storing velocities in momentum update

    def train(self, X, y, X_val, y_val,
              model, loss_function,
              reg=0.0,
              learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
              update='momentum', sample_batches=True,
              num_epochs=30, batch_size=100, acc_frequency=None,
              verbose=False, regress=True):
        """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of training data; each X[i] is a training sample.
    - y: Vector of training labels; y[i] gives the label for X[i].
    - X_val: Array of validation data
    - y_val: Vector of validation labels
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
    - num_epochs: The number of epochs to take over the training data.
    - batch_size: The number of training samples to use at each iteration.
    - acc_frequency: If set to an integer, we compute the training and
      validation set error after every acc_frequency iterations.
    - verbose: If True, print status after each epoch.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_err_history = []
        val_err_history = []
        best_val_err = float("inf")
        best_model = {}
        for it in range(num_epochs):
            X_batch = None
            y_batch = None
            random_batch = np.random.randint(0, num_train, batch_size)
            arr = []
            for i in range(X.shape[0]):
                arr.append(X[i, :, :, :])
            arr = np.array(arr)
            arr_batch = arr[random_batch]
            X_batch = []

            for n in range(arr_batch.shape[0]):
                #X_batch.append(arr_batch[n, :].reshape((1, 1, arr_batch.shape[1])))
                X_batch.append(arr_batch[n, :])
            X_batch = np.array(X_batch)
            y_batch = y[random_batch]  #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            pass
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = loss_function(X_batch, model, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for p in model:
                dx = -learning_rate * grads[p]
                model[p] += dx

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            # print(it)
            if verbose and it % 10 == 0:
                print('iteration {:d} / {:d}: loss {:f}'.format(it, num_epochs, loss))
            # Every epoch, check train and val error and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check error
                scores_train, w = loss_function(X_batch, model)
                train_err = 0.5*np.sum((scores_train - y_batch)**2)
                for n in w:
                    train_err+=0.5*reg*np.sum(n**2)
                train_err = train_err/X_batch.shape[0]
                #train_err = np.sum(np.square(scores_train - y_batch), axis=1).mean()
                scores_val, w = loss_function(X_val, model)
                val_err = 0.5*np.sum((scores_val - y_val)**2)
                for n in w:
                    val_err+=0.5*reg*np.sum(n**2)
                val_err = train_err/X_val.shape[0]
                #val_err = np.sum(np.square(scores_val - y_val), axis=1).mean()
                if val_err<best_val_err:
                    best_val_err = val_err
                    best_model = model
                train_err_history.append(train_err)
                val_err_history.append(val_err)

                # Decay learning rate
                learning_rate *= learning_rate_decay
            
        return best_val_err, best_model, loss_history, train_err_history, val_err_history

