#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import math
import numpy as np
import os
import tensorflow as tf
import tf_data

#from slim import inception_model as inception
import slim

# Logging conform @gheinrich's Torch {0,1,2,3}={INFO,WARNING,ERROR,FAIL}={.info, .warn, .error, ?}
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# Basic model parameters. #float, integer, boolean, string
tf.app.flags.DEFINE_integer('batchSize', 0, """Number of images to process in a batch""")
tf.app.flags.DEFINE_boolean('crop', False, """If this option is true, all the images are randomly cropped into square image with provided --croplen""")
tf.app.flags.DEFINE_integer('croplen', 0, """Crop length. This is required parameter when crop option is provided.""")
tf.app.flags.DEFINE_string('dbbackend', 'lmdb', """Backend of the input database""")
tf.app.flags.DEFINE_integer('epoch', 1, """Number of epochs to train, -1 for unbounded""")
tf.app.flags.DEFINE_integer('interval', 1, """Number of train epochs to complete, to perform one validation""")
tf.app.flags.DEFINE_string('labels', '', """File containing label definitions""")
tf.app.flags.DEFINE_float('learningRate', '0.001', """Learning Rate""")
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('save', 'results', """Save directory""")
tf.app.flags.DEFINE_string('seed', '', """Fixed input seed for repeatable experiments""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Shuffle records before training""")
tf.app.flags.DEFINE_integer('snapshotInterval', 1, """Specifies the training epochs to be completed before taking a snapshot""")
tf.app.flags.DEFINE_string('snapshotPrefix', '', """Prefix of the weights/snapshots""")
tf.app.flags.DEFINE_string('train', '', """Directory with training db""")
tf.app.flags.DEFINE_string('validation', '', """Directory with validation db""")

tf.app.flags.DEFINE_string('summaries_dir', '', """Directory of Tensorboard Summaries (logdir)""")



# Constants
DEFAULT_BATCH_SIZE = 16


# TIM'S OVERRIDES:
FLAGS.labels = "/Users/tzaman/Desktop/20160203-153604-2bf4/labels.txt"
FLAGS.networkDirectory = "../../digits/standard-networks/tensorflow"
FLAGS.network = "lenet_slim.py"
#FLAGS.network = "lenet.py"
FLAGS.train = "/Users/tzaman/Desktop/20160203-153604-2bf4/train_db"
FLAGS.validation = "/Users/tzaman/Desktop/20160203-153604-2bf4/val_db"
FLAGS.summaries_dir = "/tmp/tb/"
FLAGS.save = "/Users/tzaman/Desktop/result"
FLAGS.seed = 1
FLAGS.epoch = 5
FLAGS.learningRate = 0.01


# Set Tensorboard log directory
if FLAGS.summaries_dir:
    # The following gives a nice but unrobust timestamp
    FLAGS.summaries_dir = os.path.join(FLAGS.summaries_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

if FLAGS.crop and FLAGS.croplen == 0:
    logging.error("crop flag defined while croplen is 0")
    exit(-1)

if not FLAGS.train:
    logging.error("Train DB should be specified")
    exit(-1)


classes = 0
nclasses = 0
if FLAGS.labels:
    logging.info("Loading label definitions from " + FLAGS.labels + " file")
    classes = tf_data.loadLabels(FLAGS.labels)
    nclasses = len(classes)
    if not classes:
        logging.error("Reading labels file " + FLAGS.labels + " failed.")
        exit(-1)
    logging.info('Found ' + str(nclasses) + " classes")

train_data_loader = tf_data.DataLoader(FLAGS.train, nclasses, FLAGS.shuffle)

_, input_tensor_shape = train_data_loader.getInfo()
logging.info("Found " + str(train_data_loader.total) + " images in train db " + FLAGS.train)

if FLAGS.validation:
    val_data_loader = tf_data.DataLoader(FLAGS.validation, nclasses, FLAGS.shuffle)
else:
    val_data_loader = None

if FLAGS.seed:
    tf.set_random_seed(int(FLAGS.seed))
    train_data_loader.setSeed(int(FLAGS.seed))
    val_data_loader.setSeed(int(FLAGS.seed))

# update input_tensor_shape if crop length specified
# this is necessary as the input_tensor_shape is provided
# below to the user-defined function that defines the network
if FLAGS.crop and input_tensor_shape:
    input_tensor_shape[0] = FLAGS.croplen
    input_tensor_shape[1] = FLAGS.croplen


if FLAGS.dbbackend is not 'lmdb':
    logging.error("Currently only the LMDB data backend is supported")
    exit(-1)

# During training, a log output should occur at least 8 times per epoch or every 5000 images, whichever lower
logging_check_interval = math.ceil(train_data_loader.total/8) if math.ceil(train_data_loader.total/8)<5000 else 5000

logging.info("During training. details will be logged after every " + str(logging_check_interval) + " images")

# This variable keeps track of next epoch, when to perform validation.
next_validation = FLAGS.interval
logging.info("Training epochs to be completed for each validation : " + str(next_validation))
last_validation_epoch = 0

# This variable keeps track of next epoch, when to save model weights.
next_snapshot_save = FLAGS.snapshotInterval
logging.info("Training epochs to be completed before taking a snapshot : " + str(next_snapshot_save))
last_snapshot_save_epoch = 0

snapshot_prefix = FLAGS.snapshotPrefix if FLAGS.snapshotPrefix else FLAGS.network

if not os.path.exists(FLAGS.save):
    os.makedirs(FLAGS.save)
    logging.info("Created a directory " + FLAGS.save + " to save all the snapshots")


ngpus = 1 # TODO: how to handle this correctly with tf? 

# Set up input and output tensors
if input_tensor_shape[2] == 1:
    x = tf.placeholder(tf.float32, [None, input_tensor_shape[0], input_tensor_shape[1]])
else :
    x = tf.placeholder(tf.float32, [None, input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2]])
y = tf.placeholder(tf.float32, [None, nclasses])

# Import the network file
exec(open(os.path.join(FLAGS.networkDirectory, FLAGS.network)).read(), globals())

model_params= {
    'x' : x, # Input Tensor
    'y' : y, # Output Tensor
    'nclasses' : nclasses, 
    'input_shape' : input_tensor_shape, 
    'ngpus' : ngpus
}

network = build_model(model_params)

if not network.has_key('cost'):
    logging.error("Cost function definition required in model but not supplied.")
    exit(-1)

if not network.has_key('feed_dict_train'):
    network['feed_dict_train'] = {}

if not network.has_key('feed_dict_val'):
    network['feed_dict_val'] = {}

# Evaluate model
correct_pred = tf.equal(tf.argmax(network['model'], 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learningRate).minimize(network['cost'])
#lr = tf.train.exponential_decay(0.001, #initial_learning_rate
#                                0,
#                                500,
#                                0.16, # learning_rate_decay_factor
#                                staircase=True)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(network['cost']) # You can now add `feed_dict={learning_rate: 0.1}`
# Decay the learning rate exponentially based on the number of steps.
#lr = tf.train.exponential_decay(0.1, #initial_learning_rate
#                                FLAGS.epoch*train_data_loader.total,
#                                train_data_loader.total/10,
#                                0.16, # learning_rate_decay_factor
#                                staircase=True)
#
## Create an optimizer that performs gradient descent.
#RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
#RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
#RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
#optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
#                                momentum=RMSPROP_MOMENTUM,
#                                epsilon=RMSPROP_EPSILON)


# if the crop length was not defined on command line then
# check if the network defined a preferred crop length
if not FLAGS.crop and network.has_key('croplen'):
    FLAGS.crop = true
    FLAGS.croplen = network['croplen']

# TODO: do something with the croplen

# if batch size was not specified on command line then check
# whether the network defined a preferred batch size (there
# can be separate batch sizes for the training and validation
# sets)
if FLAGS.batchSize == 0:
    batch_size_train = network['train_batch_size'] if network.has_key('train_batch_size') else DEFAULT_BATCH_SIZE
    batch_size_val = network['validation_batch_size'] if network.has_key('validation_batch_size') else DEFAULT_BATCH_SIZE
else:
    batch_size_train = FLAGS.batchSize
    batch_size_val = FLAGS.batchSize

logging.info("Train batch size is " + str(batch_size_train) + " and validation batch size is " + str(batch_size_val))


# epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
epoch_round = 0 # holds the required number of significant digits for round function.
tmp_batchsize = batch_size_train
while tmp_batchsize <= train_data_loader.total:
    tmp_batchsize = tmp_batchsize * 10
    epoch_round = epoch_round + 1

logging.info("While logging, epoch value will be rounded to " + str(epoch_round) + " significant digits")

logging.info("Model weights will be saved as " + snapshot_prefix + "_<EPOCH>_Model.ckpt")

# TensorBoard
with tf.name_scope('tims_tower') as scope:
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
# Add histograms for trainable variables.
for var in tf.trainable_variables():
    summaries.append(tf.histogram_summary(var.op.name, var))
# Add a summary to track the learning rate.
#summaries.append(tf.scalar_summary('learning_rate', lr))
summary_op = tf.merge_summary(summaries)


# Saver to save all variables
saver = tf.train.Saver() 
#
#saver.restore(sess, ckpt.model_checkpoint_path)


# Launch the graph
with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir)

    # Initialize
    init = tf.initialize_all_variables().run()



    # Tensorboard: Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #merged = tf.merge_all_summaries()
    #writer_train = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    #writer_val = tf.train.SummaryWriter(FLAGS.summaries_dir + '/val')

    for epoch in xrange(0,FLAGS.epoch):
        data_loader_index = 0
        t = 0
        logged_since_last_check = 0
        while t < train_data_loader.total:

            data_batch_size = min(train_data_loader.total-data_loader_index, batch_size_train)

            batch_x, batch_y = train_data_loader.next_batch(data_batch_size, data_loader_index)

            # Update for next batch start index
            data_loader_index = data_loader_index + data_batch_size
            
            
            logged_since_last_check = logged_since_last_check + data_batch_size

            current_epoch = epoch + round(float(t)/train_data_loader.total, epoch_round)

            # Start with a forward pass
            if (t==0) or (logged_since_last_check>=logging_check_interval):
                logged_since_last_check = 0
                # Forward pass #TODO run this always..
                loss, acc = sess.run([network['cost'], accuracy], feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_val'].items() ))
                if np.isnan(loss):
                    logging.info('Model diverged with loss = NaN')
                    exit(-1)

                #writer_train.add_summary(loss, current_epoch)
                logging.info("Training (epoch " + str(current_epoch) + "): loss = " + "{:.6f}".format(loss) + ", lr = " + str(1337)  + ", accuracy = " + "{:.5f}".format(acc) )            

            # Backward pass
            sess.run(optimizer, feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_train'].items()))

            if (t%100==0):
                # TensorBoard:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, current_epoch)

            # The backward pass is completed, and we update the current epoch
            t = t + data_batch_size
            current_epoch = epoch + round(float(t)/train_data_loader.total, epoch_round)

            # Validation Pass
            if FLAGS.validation and current_epoch >= next_validation:
                #TODO: Validation()
                loss_cum_val = 0
                acc_cum_val = 0
                t_v = 0
                while t_v < val_data_loader.total:
                    data_batch_size_v = min(val_data_loader.total-t_v, batch_size_val)
                    batch_x, batch_y = val_data_loader.next_batch(data_batch_size_v, t_v)
                    loss, acc = sess.run([network['cost'], accuracy], feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_val'].items() ))
                    #loss_cum_val = loss_cum_val + loss #TODO: loss doesnt seem to be correlated to batch size???
                    acc_cum_val = acc_cum_val + (acc * data_batch_size_v)
                    t_v = t_v + data_batch_size_v

                logging.info("Validation (epoch " + str(current_epoch) + "): loss = " + "{:.6f}".format(float(loss_cum_val)/val_data_loader.total) + ", accuracy = " + "{:.5f}".format(acc_cum_val/val_data_loader.total) )




                #To find next nearest epoch value that exactly divisible by FLAGS.interval:
                next_validation = (round(float(current_epoch)/FLAGS.interval) + 1) * FLAGS.interval 
                last_validation_epoch = current_epoch

            # Saving Snapshot
            if current_epoch >= next_snapshot_save:
                filename_snapshot = os.path.join(FLAGS.save, snapshot_prefix + "_" + str(current_epoch) + "_Model.ckpt")
                logging.info("Snapshotting to " + filename_snapshot)
                saver.save(sess, filename_snapshot)
                logging.info("Snapshot saved - " + filename_snapshot)

                # To find next nearest epoch value that exactly divisible by FLAGS.snapshotInterval
                next_snapshot_save = (round(float(current_epoch)/FLAGS.snapshotInterval) + 1) * FLAGS.snapshotInterval 
                last_snapshot_save_epoch = current_epoch



            
            


