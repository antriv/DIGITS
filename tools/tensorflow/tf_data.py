# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

"""Defines data loading and processing"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("/Users/tzaman/caffe/python/caffe/proto/") #@TODO: REMOVEME
import caffe_pb2
import lmdb
import tensorflow as tf
import numpy as np
import random

import PIL.Image
from StringIO import StringIO

def loadLabels(filename):
    with open(filename) as f:
        return f.readlines()

class DataLoader:

    def lmdb_getSample(self, shuffle, idx):
        #print("lmdb_getSample(shuffle=" + str(shuffle) + ", idx=" + str(idx) + ")")
        if shuffle:
            idx = random.randint(0,self.total-1)

        im_db = self.lmdb_txn.get(self.keys[idx])

        datum = caffe_pb2.Datum()
        datum.ParseFromString(im_db)
        label = datum.label

        label_onehot = np.zeros(self.nclasses, dtype=int)
        label_onehot[int(label)] = 1
    
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        im_pil = PIL.Image.open(s)
        im_np = np.asarray(im_pil)

        #return im_np, label_onehot # One-Hot Approach
        return im_np, label

    def getFirstImageShape(self):
        # Sample first image to get dimensions
        im_db = self.lmdb_txn.get(self.keys[0])
        datum = caffe_pb2.Datum()
        datum.ParseFromString(im_db)
        # Note: tensorFlow shape convention: [h,w,c]
        return [datum.height, datum.width, datum.channels] 

    def getInfo(self):
        return self.total, self.input_tensor_shape

    def next_batch(self, batch_size, idx):
        #print "next_batch( batch_size=" + str(batch_size) + ", idx=" + str(idx)

        if self.input_tensor_shape[2] == 1:
            images = np.empty([batch_size, self.input_tensor_shape[1], self.input_tensor_shape[0]], dtype=float)
        else:
            images = np.empty([batch_size, self.input_tensor_shape[1], self.input_tensor_shape[0], self.input_tensor_shape[2]], dtype=float)
        #labels = np.empty([batch_size, self.nclasses], dtype=int) # One-Hot Approach
        labels = np.empty([batch_size], dtype=int)

        for i in xrange(0, batch_size):
            # Get next single sample
            image, label = self.lmdb_getSample(self.shuffle, idx+i)
            images[i] = image
            labels[i] = label

        # Keep data in DIGITS [0 255] default range
        return (images-127.5)/127.5, labels 

    def __init__(self, location, nclasses, shuffle):
        print("DataLoader __init__()")

        # Set up the data loader
        self.lmdb_env = lmdb.open(location, readonly=True, lock=False)
        
        self.lmdb_txn = self.lmdb_env.begin(buffers=False)
        self.total = self.lmdb_txn.stat()['entries']
        self.keys = []
        for key, value in self.lmdb_txn.cursor():
            self.keys.append(key)

        # Set options
        self.shuffle = shuffle
        self.nclasses = nclasses

        # Obtain information
        self.input_tensor_shape = self.getFirstImageShape()

    def setSeed(self, seed):
        # The following should be removed once we get rid of numpy
        random.seed(seed)












