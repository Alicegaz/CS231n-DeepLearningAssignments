import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M). out = x w + b
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # Affine forward pass. First reshape the input into rows such that it has   #
  # shape N x D.                                                              #
  #############################################################################
  out = np.dot(x.reshape((x.shape[0], w.shape[0])), w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer (given in the cache).

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)


  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # Affine backward pass.                                                     #
  #############################################################################
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape((x.shape[0], w.shape[0])).T, dout)
  d = [[1, 3, 5], [2, 4, 3]]
  #print(dout.shape)
  #print(np.sum(d, axis=0))
  db = np.sum(dout, axis = 0)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  # ReLU forward pass.                                                        #
  #############################################################################
  out = np.maximum(0, x)
  cache = x

  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  #############################################################################
  # ReLU backward pass.                                                       #
  #############################################################################

  dx = dout
  dx[cache<0] = 0
  return dx

def leaky_relu_forward(x, p = 0.01):
    out = x
    #print("be", out)
    out[x < 0]=out[x<0]*p
    #print("af", out)
    cache = (x, p)
    return out, cache

def leaky_relu_backward(dout, cache):
    x, p = cache
    dx = dout
    #print("back x", x)
    #print("dx", dx)
    dx[x<0] = dx[x<0]*p
    return dx
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  #############################################################################
  # Convolutional forward pass                                                #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  H_pad = H + 2*pad
  W_pad = W + 2*pad
  Hp = int(1 + (H + 2 * pad - HH) / stride)
  Wp = int(1 + (W + 2 * pad - WW) / stride)
  
  out = np.zeros((N, F, Hp, Wp))

  # Add padding around each 2D image
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    
  for i in range(N): # ith example
    for j in range(F): # jth filter
      # Convolve this filter over windows
      k = 0
      hs = 0
      while hs+HH-1<= H_pad-1:
        l = 0
        ws = 0
        while ws+WW-1<=W_pad-1:
          # Convolve the jth filter over (C, HH, WW)
          # 
          # FILL IN THIS PART
          # 
          #print("hs:hs+HH", hs, hs+HH, "ws:ws+WW", ws, ws+WW, "im", image[:, hs:hs+HH, ws:ws+WW])
          out[i, j, k, l] = np.sum(padded[i, :, hs:hs+HH, ws:ws+WW]*w[j]) + b[j]
          ws = ws+stride
 
          l=l+1
          
        k=k+1
        hs = hs + stride
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  #############################################################################
  # Convolutional backward pass                                               #
  #############################################################################
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = int(1 + (H + 2 * pad - HH) / stride)
  Wp = int(1 + (W + 2 * pad - WW) / stride)
  H_pad = H + 2*pad
  W_pad = W + 2*pad
  #dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
  dx = np.zeros_like(padded)
  # You may need to pad the gradient and then unpad
  db = np.sum(np.sum(np.sum(dout, axis=0), axis=1), axis=1)
  
  for i in range(N): # ith example
    for j in range(F): # jth filter
      # Convolve this filter over windows
      hs = 0
      l = 0
      while hs+HH-1 <= H_pad-1:
        k = 0
        ws = 0
        while ws+WW-1 <= W_pad-1:
          # Compute gradient of out[i, j, k, l] = np.sum(window*w[j]) + b[j]
          # 
          # FILL IN THIS PART 
          dw[j, :, :, :] += padded[i, :, hs:hs+HH, ws:ws+WW]*dout[i, j, l, k]
          dx[i, :, hs:hs+HH, ws:ws+WW] += np.dot(w[j], dout[i, j, l, k])
          ws += stride
          k+=1
        hs = hs + stride
        l+=1
  dx = dx[:, :, pad:-pad, pad:-pad]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  #############################################################################
  # Max pooling forward pass                                                  #
  #############################################################################
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = int(1 + (H - HH) / stride)
  Wp = int(1 + (W - WW) / stride)
  #print("x", x.shape, "s", stride, WW, HH)
  out = np.zeros((N, C, Hp, Wp))
  #print(x)
  for i in range(N):
    # Need this; apparently we are required to max separately over each channel
    for j in range(C):
      k = 0
      hs = 0
      while hs+HH-1 <= H:
        l = 0
        ws = 0
        while ws+WW-1 <= W:
          #print("WW", WW, "HH", HH)
          #print("im", ws+WW, hs+HH, x[i, j, hs:hs+HH, ws:ws+WW])
          out[i, j, k, l] = np.max(x[i, j, hs:hs+HH, ws:ws+WW])
          #print("out", x[i, j, hs:hs+HH, ws:ws+WW], np.max(x[i, j, hs:hs+HH, ws:ws+WW]))
          # 
          # FILL IN THIS PART 
          # 
          #print(l, ws+Wp-1, Wp)
          ws = ws+stride
          l += 1
          #print("end", ws+WW-1, Wp)
        hs = hs+stride
        k+=1

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """

  #############################################################################
  # Max pooling backward pass                                                 #
  #############################################################################
  x, pool_param = cache
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = int(1 + (H - HH) / stride)
  Wp = int(1 + (W - WW) / stride)

  dx = np.zeros_like(x)

  for n in range(N):
    for c in range(C):
      for i in range(Hp):
        for j in range(Wp):
          #dx[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW][x_window != x_max] = 0 
          dx[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]+=(x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW] == np.max(x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW])) * dout[n, c, i, j]

  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def ser_loss(x, y, w, reg = 0.5):
    loss = 0.5*np.sum((x-y)**2)
    for n in w:
        loss+=0.5*reg*np.sum(n**2)
    dx = x - y
    return loss/x.shape[0], dx/x.shape[0]

def huber_loss_class(x, y, d=1.00):
    N = x.shape[0]
    y = x[np.arange(N), y]
    l = np.sum((y - x)**2)
    if(l<=d):
        dx = (x - y)/x.shape[0]
        return (0.5*np.sum((x-y)**2))/(x.shape[0]), dx
    else:
        dx = 2*d*(x-y)/x.shape[0]
        return (d*np.sum((x-y)**2) - 0.5*d**2)/(x.shape[0]), dx


def huber_loss(x, y, d=1.00):
    l = np.sum((y - x)**2)
    if(l<=d):
        dx = (x - y)/x.shape[0]
        return (0.5*np.sum((x-y)**2))/(x.shape[0]), dx
    else:
        dx = 2*d*(x-y)/x.shape[0]
        return (d*np.sum((x-y)**2) - 0.5*d**2)/(x.shape[0]), dx
    
def pseudo_huber_loss(x, y, d=1.00, m = 1.00):
    dx = (1+(x - y))/y.shape[0]
    return d*np.sum(np.sqrt(m**2+(x-y)**2)-m)/x.shape[0], dx

def pseudo_huber_loss_class(x, y, d=1.00, m = 1.00):
    N = x.shape[0]
    y = x[np.arange(N), y]
    dx = (1+(x - y))/y.shape[0]
    return d*np.sum(np.sqrt(m**2+(x-y)**2)-m)/x.shape[0], dx

def softmax_loss_unnorm(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, x[np.arange(N), y]

def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    # Compute output
    mu = x.mean(axis=0)
    xc = x - mu
    var = np.mean(xc ** 2, axis=0)
    std = np.sqrt(var + eps)
    xn = xc / std
    out = gamma * xn + beta

    cache = (mode, x, gamma, xc, std, xn, out)

    # Update running average of mean
    running_mean *= momentum
    running_mean += (1 - momentum) * mu

    # Update running average of variance
    running_var *= momentum
    running_var += (1 - momentum) * var
  elif mode == 'test':
    # Using running mean and variance to normalize
    std = np.sqrt(running_var + eps)
    xn = (x - running_mean) / std
    out = gamma * xn + beta
    cache = (mode, x, xn, gamma, beta, std)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  mode = cache[0]
  if mode == 'train':
    mode, x, gamma, xc, std, xn, out = cache

    N = x.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dxc = dxn / std
    dstd = -np.sum((dxn * xc) / (std * std), axis=0)
    dvar = 0.5 * dstd / std
    dxc += (2.0 / N) * xc * dvar
    dmu = np.sum(dxc, axis=0)
    dx = dxc - dmu / N
  elif mode == 'test':
    mode, x, xn, gamma, beta, std = cache
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dx = dxn / std
  else:
    raise ValueError(mode)

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape
  x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
  out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
  out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = dout.shape
  dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
  dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = None
    out = None
    
    if mode == 'train':
        mask = (np.random.random(x.shape)>p)/(1-p)
        out = x * mask
    elif mode == 'test':
        out = x
        mask = None
    cache = (dropout_param, out)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']
    if mode =='train':
        dX = dout*mask
    elif mode == 'test':
        dX=dout
    return dX
