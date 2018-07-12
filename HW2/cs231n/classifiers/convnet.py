import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape
   
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads

def three_layer_convnet(X, model, y=None, reg=0.0):
  W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
  N, C, H, W = X.shape
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
  conv_filter_height1, conv_filter_width1 = W2.shape[2:]
  assert conv_filter_height1 == conv_filter_width1, 'Conv filter must be square'
  assert conv_filter_height1 % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width1 % 2 == 1, 'Conv filte r width must be odd'
  conv_param1 = {'stride': 1, 'pad': (conv_filter_height1 - 1) / 2}
  pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
  scores, cache3 = affine_forward(a2, W3, b3)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da2, dW3, db3 = affine_backward(dscores, cache3)
  da1,  dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
  return loss, grads

def four_layer_convnet(X, model, y=None, reg=0.0):
  """
   3*(ConV+Relu+Pooling)+affine
  """
  W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
  N, C, H, W = X.shape
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
  a3, cache3 = conv_relu_pool_forward(a2, W3, b2, conv_param, pool_param)
  scores, cache4 = affine_forward(a3, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)
  #data_loss, dscores = huber_loss_class(scores, y)
  #data_loss, dscores = pseudo_huber_loss_class(scores, y)
  # Compute the gradients using a backward pass
  da3, dW4, db4 = affine_backward(dscores, cache4)
  da2,  dW3, db3 = conv_relu_pool_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}
  return loss, grads

def sup_layer_convnet(X, model, y=None, reg=0.0):
  """
   3*(ConV+Relu+Pooling)+affine
  """
  mode = 'train'
  if y is None:
    mode = 'test'
  W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'], model['W5'], model['b5'], model['W6'], model['b6'], model['W7'], model['b7']
  N, C, H, W = X.shape
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
  dropout_param = {'p': 0.25, 'mode': mode, 'seed': 123}
  # Compute the forward pass
  a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  a1d, cache2 = dropout_forward(a1, dropout_param)
  a2, cache3 = conv_relu_forward(a1d, W2, b2, conv_param)
  a2d, cache4 = dropout_forward(a2, dropout_param)
  a3, cache5 = conv_relu_forward(a2d, W3, b2, conv_param)
  a3d, cache6 = dropout_forward(a3, dropout_param) 
  a4, cache7 = conv_relu_pool_forward(a3d, W4, b4, conv_param, pool_param)
  a4d, cache8 = dropout_forward(a4, dropout_param)
  a5, cache9 = conv_relu_pool_forward(a4d, W5, b5, conv_param, pool_param)
  a5d, cache10 = dropout_forward(a5, dropout_param)
  a6, cache11 = conv_relu_pool_forward(a5d, W6, b6, conv_param, pool_param)
  a6d, cache12 = dropout_forward(a6, dropout_param) 
  scores, cache13 = affine_forward(a6d, W7, b7)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da6, dW7, db7 = affine_backward(dscores, cache13)
  da6o = dropout_backward(da6, cache12)
  da5,  dW6, db6 = conv_relu_pool_backward(da6o, cache11)
  da5o = dropout_backward(da5, cache10)
  da4, dW5, db5 = conv_relu_pool_backward(da5o, cache9)
  da4o = dropout_backward(da4, cache8)
  da3, dW4, db4 = conv_relu_pool_backward(da4o, cache7)
  da3o = dropout_backward(da3, cache6)
  da2, dW3, db3 = conv_relu_backward(da3o, cache5)
  da2o = dropout_backward(da2, cache4)
  da1, dW2, db2 = conv_relu_backward(da2o, cache3)
  da1o = dropout_backward(da1, cache2)
  dX, dW1, db1 = conv_relu_backward(da1o, cache1)
   
  

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  dW5 += reg * W5
  dW6 += reg * W6
  dW7 += reg * W7
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6, W7])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7}
  return loss, grads



def advlayer_net(X, model, y=None, reg=0.0, mode='train'):
  """
   3*(ConV+Relu+Pooling)+affine
  """
  W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
  N, C, H, W = X.shape
  X = X.round(8)
  conv_filter_height, conv_filter_width = W1.shape[2:]
  #assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 2, 'pad': 0}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
  dropout_param = {'p': 0.25, 'mode': mode, 'seed': 123}
  # Compute the forward pass
  #a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  #print(a1[1, 0, 0, :30])
  #a2, cache2 = dropout_forward(a1, dropout_param)
  #print(a2[1, 0, 0, :30])
  #a3, cache3 = conv_relu_forward(a2, W2, b2, conv_param)
  #a4, cache4 = dropout_forward(a3, dropout_param)
  #a5, cache5 = conv_relu_forward(a4, W3, b2, conv_param)
  #a6, cache6 = dropout_forward(a5, dropout_param)
  #scores, cache7 = affine_forward(a6, W4, b4)
  a1, cache1 = conv_leaky_relu_forward(X, W1, b1, conv_param)
  #print("a1",a1[1, 0, 0, :30])
  a2, cache2 = conv_leaky_relu_forward(a1, W2, b2, conv_param)
  #print("a2",a2[1, 0, 0, :30])
  a3, cache3 = conv_leaky_relu_forward(a2, W3, b3, conv_param)
  #print("a3",a3)
  scores, cache4 = affine_forward(a3, W4, b4)
  scores = scores.round(8)
  w = np.array([W1, W2, W3, W4])
  if y is None:
    return scores.round(8), w

  
  # Compute the backward pass
  data_loss, dscores = ser_loss(scores, y, w, reg=reg)
  #print("scores", scores[:20, :20])
  # Compute the gradients using a backward pass
  #da6, dW4, db4 = affine_backward(dscores, cache7)
  #da5 = dropout_backward(da6, cache6)
  #da4,  dW3, db3 = conv_relu_backward(da5, cache5)
  #da3 = dropout_backward(da4, cache4)
  #da2, dW2, db2 = conv_relu_backward(da3, cache3)
  #da1 = dropout_backward(da2, cache2)
  da3, dW4, db4 = affine_backward(dscores, cache4)
  da2, dW3, db3 = conv_leaky_relu_backward(da3, cache3)
  da1, dW2, db2 = conv_leaky_relu_backward(da2, cache2)
  dX, dW1, db1 = conv_leaky_relu_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}
  return loss, grads


def pool_layer_convnet(X, model, y=None, reg=0.0):
  """
   3*(ConV+Relu+Pooling)+affine
  """
  W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'], model['W5'], model['b5'], model['W6'], model['b6'], model['W7'], model['b7'], model['W8'], model['b8']
  N, C, H, W = X.shape
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
  # Compute the forward pass
  a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  a2, cache2 = conv_relu_forward(a1, W2, b2, conv_param)
  a3, cache3 = conv_relu_forward(a2, W3, b3, conv_param)
  a4, cache4 = conv_relu_forward(a3, W4, b4, conv_param)
  a5, cache5 = conv_relu_forward(a4, W5, b5, conv_param)
  a6, cache6 = conv_relu_pool_forward(a5, W6, b6, conv_param, pool_param)
  a7, cache7 = conv_relu_pool_forward(a6, W7, b7, conv_param, pool_param)
  scores, cache8 = affine_forward(a7, W8, b8)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da7, dW8, db8 = affine_backward(dscores, cache8)
  da6,  dW7, db7 = conv_relu_pool_backward(da7, cache7)
  da5, dW6, db6 = conv_relu_pool_backward(da6, cache6)
  da4, dW5, db5 = conv_relu_backward(da5, cache5)
  da3,  dW4, db4 = conv_relu_backward(da4, cache4)
  da2, dW3, db3 = conv_relu_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  dW5 += reg * W5
  dW6 += reg * W6
  dW7 += reg * W7
  dW8 += reg * W8
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6, W7, W8])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7, 'W8': dW8, 'b8': db8}
  return loss, grads

def affin_layer_convnet(X, model, y=None, reg=0.0):
  """
   3*(ConV+Relu+Pooling)+affine
  """
  W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'], model['W5'], model['b5']
  N, C, H, W = X.shape
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
  # Compute the forward pass
  a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  a2, cache2 = conv_relu_forward(a1, W2, b2, conv_param)
  a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param, pool_param)
  a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param, pool_param)
  scores, cache5 = affine_forward(a4, W5, b5)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da4, dW5, db5 = affine_backward(dscores, cache5)
  da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
  da2, dW3, db3 = conv_relu_pool_backward(da3, cache3)
  da1,  dW2, db2 = conv_relu_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  dW5 += reg * W5
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5}
  return loss, grads





def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * int(H * W / 4), num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model

def init_three_layers_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5, stride=True):
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
    
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  if (stride):
      model['W3'] = weight_scale * np.random.randn(num_filters*int(H*W/16), num_classes)
  else:
      model['W3'] = weight_scale * np.random.randn(num_filters*H*W, num_classes)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_classes)
  return model

def init_four_layer_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(num_filters*int( (H/(2**(3)))*(W/(2**(3)))), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model

def init_sup_layer_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W5'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W6'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W7'] = weight_scale * np.random.randn(num_filters*int( (H/(2**(3)))*(W/(2**(3)))), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_filters)
  model['b5'] = bias_scale * np.random.randn(num_filters)
  model['b6'] = bias_scale * np.random.randn(num_filters)
  model['b7'] = bias_scale * np.random.randn(num_classes)
  return model

def init_advlayer_net(weight_scale=1e-5, bias_scale=0, input_shape=(1, 32, 32),
                           num_classes=2, num_filters=32, filter_size=5, stride=False):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  stride = True
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, 1, filter_size)
  
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, 1, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, 1, filter_size)
  if stride:
        model['W4'] = weight_scale * np.random.randn(num_filters*int((1)*((W/(2**3))-1)), num_classes)
  else:
      model['W4'] = weight_scale * np.random.randn(num_filters*int(H*W), num_classes)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model

def init_pool_layer_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W5'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W6'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W7'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W8'] = weight_scale * np.random.randn(num_filters*int( (H/(2**(2)))*(W/(2**(2)))), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_filters)
  model['b5'] = bias_scale * np.random.randn(num_filters)
  model['b6'] = bias_scale * np.random.randn(num_filters)
  model['b7'] = bias_scale * np.random.randn(num_filters)
  model['b8'] = bias_scale * np.random.randn(num_classes)
  return model


def init_four_layer_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(num_filters*int( (H/(2**(3)))*(W/(2**(3)))), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model

def init_affin_layer_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W5'] = weight_scale * np.random.randn(num_filters*int( (H/(2**(2)))*(W/(2**(2)))), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_filters)
  model['b5'] = bias_scale * np.random.randn(num_classes)
  return model