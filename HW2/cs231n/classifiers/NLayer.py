import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def n_layer_cnn_init(weight_scale=1e-3, bias_scale=0, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 num_filters_conv=20, num_classes=10, num_hidden_conv=3):
        C, H, W = input_dim
        model = {}
        model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        model['b1'] = bias_scale * np.random.randn(num_filters)
        for i in range(num_hidden_conv - 2):
            model['W' + str(i + 2)] = weight_scale * np.random.randn(num_filters_conv, num_filters, filter_size,
                                                                     filter_size)
            model['b' + str(i + 2)] = bias_scale * np.random.randn(num_filters_conv)

        model['W'+str(i+3)] = weight_scale * np.random.randn(
            num_filters * int((H / (2 ** num_hidden_conv)) * (W / (2 ** num_hidden_conv)) / 4), num_classes)
        model['b'+str(i+3)] = bias_scale * np.random.randn(num_classes)
        return model

def n_layer_cnn(X, model, y=None, reg=0.0):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1']
        N, C, H, W = X.shape
        conv_filter_height, conv_filter_width = W1.shape[2:]
        assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
        assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
        assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
        conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # Compute the forward pass
        last_ind = len(model)/2
        cache = []
        a = np.empty((1, last_ind))
        WW = np.empty((1, last_ind+1))
        B = np.empty((1, last_ind+1))
        #start from 1 index, a[0], cache[0] empty
        a[1], cache[1] = conv_relu_pool_forward(X, model['W1'], model['b1'], conv_param, pool_param)
        for i in range(len(model)/2-3):
                w = model['W'+str(i+2)]
                WW.append(w)
                b = model['b'+str(i+2)]
                B.append(b)
                a[i+2], cache[i+2] = conv_relu_pool_forward(a[i+1], w, b, conv_param, pool_param)

        #a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
        scores, cache[i+3] = affine_forward(a[i+2], model['W'+str(i+3)], model['b'+str(i+3)])


        if y is None:
            return scores

            # Compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)

        # Compute the gradients using a backward pass
        da = np.empty((1, (len(model)/2)))
        dW = np.empty((1, (len(model)/2)+1))
        db = np.empty((1, (len(model)/2)+1))

        da[last_ind-1], dW[last_ind], db[last_ind] = affine_backward(dscores, cache[i+3])
        for i in reversed(range(last_ind-2)):
                da[i], dW[i+1], db[i+1] = conv_relu_pool_backward(da[i+1], cache[i+1])
        for i in len(da):
            dW[i+1]+=reg*W[i+1]

        #da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
        #dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

        # Add regularization
        #dW1 += reg * W1
        #dW2 += reg * W2
        #dW3 += reg * W3
        reg_loss = 0.5 * reg * sum(np.sum(w * w) for w in WW)

        loss = data_loss + reg_loss
        grads = {}
        for i in range(len(dW)):
                grads['W'+str(i+1)] = dW[i+1]
                grads['b'+str(i+1)] = db[i+1]
        #grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
        return loss, grads


