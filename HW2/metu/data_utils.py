import scipy.io as sio # you may use scipy for loading your data
#import cPickle as pickle # or Pickle library
import numpy as np
import os
import h5py
import scipy.signal as signal
#import analytic_wfm as AW
#from peakdetect import *
import peakutils
from peakutils import *

def load_dataset(filename, input_size, min_dist):
    with h5py.File(filename, 'r') as f:
        print(list(f.keys()))
        Part = f['Part_1']
        dataset = np.array(f[Part[0].item()].value)
        f.close()
    t = 0
    W = input_size
    i = 0
    stride = 5
    size = int((dataset.shape[0]-W)/stride + 1)
    X = np.zeros((size, W))
    y = np.zeros((size, 2))
    #find peaks in array of APB in the array 
    while t<=(dataset.shape[0]-(dataset.shape[0] % stride) - W):
        print("dd", t, W+t)
        X[i, :] = np.array(dataset[t:W+t, 0]).reshape(W, 1).T 

        #up = np.percentile(X[i, :], 95)
        #down =np.percentile(X[i, :], 5)
        #x = X[i, :]
        #x[x>up] = up
        #x[x<down] = down
        #N = x.shape[0]
        #x/=x.max()
        #X[i, :] = x
        abp = np.array(np.ravel(dataset[t:W+t, 1].reshape(W, 1).T))
        #local maxima
        max_peaks = peakutils.indexes(abp, min_dist=min_dist)
        #local minima
        abp1 =1./abp
        min_peaks = peakutils.indexes(abp1, min_dist=min_dist)
        #means
        minp = np.mean([abp[k] for k in min_peaks])
        maxp = np.mean([abp[k] for k in max_peaks])
        y[i, :] = [maxp, minp]
        #t+=W
        t+=5
        i+=1
    return X, y



if __name__ == '__main__':
	# TODO: You can fill in the following part to test your function(s)/dataset from the command line
	filename='...'
	X, Y = load_dataset(filename)

