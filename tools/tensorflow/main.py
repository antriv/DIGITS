#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""TensorFlow universal-ish training executable for DIGITS
Defines the training procedure

Usage:
See the self-documenting flags below.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import datetime
import logging
import math
import numpy as np
import os
import re
import tensorflow as tf
import time


# Local imports
# Note slim will become part of Tensorflow Contrib: https://github.com/tensorflow/models/issues/203
import slim 
import tf_data

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
tf.app.flags.DEFINE_float('learningRateDecay', 1e-6, """Learning rate decay (in # samples)""") # @TODO: implement
tf.app.flags.DEFINE_float('momentum', '0.9', """Momentum""") # @TODO: implement
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('optimization', 'sgd', """Optimization method""")
#tf.app.flags.DEFINE_string('policy', '', """Learning Rate Policy""")
tf.app.flags.DEFINE_string('save', 'results', """Save directory""")
tf.app.flags.DEFINE_string('seed', '', """Fixed input seed for repeatable experiments""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Shuffle records before training""")
tf.app.flags.DEFINE_integer('snapshotInterval', 1.0, """Specifies the training epochs to be completed before taking a snapshot""")
tf.app.flags.DEFINE_string('snapshotPrefix', '', """Prefix of the weights/snapshots""")
tf.app.flags.DEFINE_string('train', '', """Directory with training db""")
tf.app.flags.DEFINE_string('validation', '', """Directory with validation db""")
tf.app.flags.DEFINE_float('weightDecay', 1e-4, """L2 penalty on the weights""") # @TODO: implement
tf.app.flags.DEFINE_string('weights', '', """Filename for weights of a model to use for fine-tuning""") # @TODO: implement


# Tensorflow-unique arguments for DIGITS
# 'tf_summaries_dir' default is '' which defaults to the cwd (jobs dir)
tf.app.flags.DEFINE_string('tf_summaries_dir', '', """Directory of Tensorboard Summaries (logdir)""") 
tf.app.flags.DEFINE_boolean('tf_serving_export', False, """Flag for exporting an Tensorflow Serving model""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# Constants
DEFAULT_BATCH_SIZE = 16
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999 # The decay for the moving average of the summaries


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
               of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    #labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in the model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def Validation(sess, x, y, val_data_loader, writer_val, batch_size_val, loss, accuracy, network, summary_op, step, current_epoch):
    #@TODO: put in class (and reduce args)
    loss_cum_val = 0
    acc_cum_val = 0
    t_v = 0
    num_batches = 0
    while t_v < val_data_loader.total:
        data_batch_size_v = min(val_data_loader.total-t_v, batch_size_val)
        batch_x, batch_y = val_data_loader.next_batch(data_batch_size_v, t_v)
        loss_val, acc_val, summary_str = sess.run([loss, accuracy, summary_op], feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_val'].items()))
        writer_val.add_summary(summary_str, step)
        loss_cum_val = loss_cum_val + loss_val
        acc_cum_val = acc_cum_val + (acc_val * data_batch_size_v) #TODO: obtain this from a confmat struct-ish thing
        t_v = t_v + data_batch_size_v
        num_batches = num_batches + 1

    total_avg_loss = float(loss_cum_val)/num_batches
    total_acc = acc_cum_val/val_data_loader.total

    logging.info("Validation (epoch " + str(current_epoch) + "): loss = " + "{:.6f}".format(total_avg_loss) + ", accuracy = " + "{:.5f}".format(total_acc))


def main(_):

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Set Tensorboard log directory
        if FLAGS.tf_summaries_dir:
            # The following gives a nice but unrobust timestamp
            FLAGS.tf_summaries_dir = os.path.join(FLAGS.tf_summaries_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        if FLAGS.crop and FLAGS.croplen == 0:
            logging.error("crop flag defined while croplen is 0")
            exit(-1)

        if not FLAGS.train:
            logging.error("Train DB should be specified")
            exit(-1)

        classes = 0
        nclasses = 0
        if FLAGS.labels:
            logging.info("Loading label definitions from %s file", FLAGS.labels)
            classes = tf_data.loadLabels(FLAGS.labels)
            nclasses = len(classes)
            if not classes:
                logging.error("Reading labels file %s failed.", FLAGS.labels)
                exit(-1)
            logging.info("Found %s classes", nclasses)

        train_data_loader = tf_data.DataLoader(FLAGS.train, nclasses, FLAGS.shuffle)

        _, input_tensor_shape = train_data_loader.getInfo()
        logging.info("Found %s images in train db %s ", train_data_loader.total, FLAGS.train)

        if FLAGS.validation:
            val_data_loader = tf_data.DataLoader(FLAGS.validation, nclasses, FLAGS.shuffle)
        else:
            val_data_loader = None


        # update input_tensor_shape if crop length specified
        # this is necessary as the input_tensor_shape is provided
        # below to the user-defined function that defines the network
        if FLAGS.crop and input_tensor_shape:
            input_tensor_shape[0] = FLAGS.croplen
            input_tensor_shape[1] = FLAGS.croplen

        if FLAGS.dbbackend != 'lmdb':
            logging.error("Currently only the LMDB data backend is supported")
            exit(-1)

        # During training, a log output should occur at least 8 times per epoch or every 5000 images, whichever lower
        logging_check_interval = math.ceil(train_data_loader.total/8) if math.ceil(train_data_loader.total/8)<5000 else 5000

        logging.info("During training. details will be logged after every %s images", logging_check_interval)

        # This variable keeps track of next epoch, when to perform validation.
        next_validation = FLAGS.interval
        logging.info("Training epochs to be completed for each validation : %s", next_validation)
        last_validation_epoch = 0

        # This variable keeps track of next epoch, when to save model weights.
        next_snapshot_save = FLAGS.snapshotInterval
        logging.info("Training epochs to be completed before taking a snapshot : %s", next_snapshot_save)
        last_snapshot_save_epoch = 0

        snapshot_prefix = FLAGS.snapshotPrefix if FLAGS.snapshotPrefix else FLAGS.network

        if not os.path.exists(FLAGS.save):
            os.makedirs(FLAGS.save)
            logging.info("Created a directory %s to save all the snapshots", FLAGS.save)


        ngpus = 1 # TODO: how to handle this correctly with tf? 

        # Set up input tensor
        if input_tensor_shape[2] == 1:
            x = tf.placeholder(tf.float32, shape=(None, input_tensor_shape[0], input_tensor_shape[1]))
        else:
            x = tf.placeholder(tf.float32, shape=(None, input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2]))

        # Set up output (truth) tensor
        #y = tf.placeholder(tf.float32, shape=(None, nclasses)) # One-Hot Approach
        y = tf.placeholder(tf.int64, shape=(None))
        #y = tf.cast(y, tf.int64) # Cast to this for classification (or previously allocate as such)

        # Import the network file
        path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.networkDirectory, FLAGS.network)
        exec(open(path_network).read(), globals())

        model_params= {
            'x' : x, # Input Tensor
            'y' : y, # Output Tensor (Truth)
            'nclasses' : nclasses, 
            'input_shape' : input_tensor_shape, 
            'ngpus' : ngpus
        }

        # Run the user model through the build_model function that should be filled in
        network = build_model(model_params)

        if not network.has_key('cost'):
            logging.error("Cost function definition required in model file but not supplied.")
            exit(-1)

        if not network.has_key('feed_dict_train'):
            network['feed_dict_train'] = {}

        if not network.has_key('feed_dict_val'):
            network['feed_dict_val'] = {}

        # The network['model'] is the op that results in the logits; (inference graph)
        if not network.has_key('model'):
            logging.error("Model definition required in model file but not supplied.")
            exit(-1)


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

        logging.info("Train batch size is %s and validation batch size is %s", batch_size_train, batch_size_val)


        # epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
        epoch_round = 0 # holds the required number of significant digits for round function.
        tmp_batchsize = batch_size_train
        while tmp_batchsize <= train_data_loader.total:
            tmp_batchsize = tmp_batchsize * 10
            epoch_round = epoch_round + 1

        logging.info("While logging, epoch value will be rounded to %s significant digits", epoch_round)

        logging.info("Model weights will be saved as %s_<EPOCH>_Model.ckpt", snapshot_prefix)

        logging.info('started training the model')

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        if FLAGS.seed:
            tf.set_random_seed(int(FLAGS.seed))
            train_data_loader.setSeed(int(FLAGS.seed))
            val_data_loader.setSeed(int(FLAGS.seed))

        # Accuracy Op
        # @TODO: check if the network here is right
        correct_pred = tf.equal(tf.argmax(network['model'], 1), y) # Equal equates to boolean. Argmax gets the index of the max.
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


         ## TensorBoard
        with tf.name_scope('tims_tower') as scope:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        cross_entropy_mean = tf.reduce_mean(network['cost'], name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = _add_loss_summaries(loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            # @TODO: Create a LRPolicy Class
            logging.info('Optimizer:' + FLAGS.optimization)
            if FLAGS.optimization == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learningRate)#.minimize(network['cost']) #FIXME: 
            elif FLAGS.optimization == 'sgd':
                print('using sgd')
                #lr = tf.train.exponential_decay(FLAGS.learningRate,
                #                        global_step,
                #                        10000,  #decay_steps
                #                        FLAGS.learningRateDecay,
                #                        staircase=True)
                #lr = tf.placeholder(tf.float32, shape=[]) # You can add lr to the feed_dict this way
                #lr = tf.Variable(FLAGS.learningRate)
                # Create an optimizer that performs gradient descent.
                optimizer = tf.train.GradientDescentOptimizer(FLAGS.learningRate)
            elif FLAGS.optimization == 'rmsprop':
                #@TODO
                # lr =
                optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)
                exit(-1)
            else:
                logging.error("Invalid optimization flag %s", FLAGS.optimization)
                exit(-1)
            grads = optimizer.compute_gradients(loss)


        # Apply gradients.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # Create a saver.
        # Will only save sharded if we want to use the model for serving
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0, sharded=FLAGS.tf_serving_export) 


        ## TensorBoard
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))
        # Add a summary to track the learning rate.
        #summaries.append(tf.scalar_summary('learning_rate', lr))
        summaries.append(tf.scalar_summary('loss', network['cost']))
        summaries.append(tf.scalar_summary('accuracy', accuracy))

        # Build the summary operation from the (last tower) summaries.
        summary_op = tf.merge_summary(summaries)


        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, # will automatically do non-gpu supported ops on cpu
            log_device_placement=FLAGS.log_device_placement))

        # Initialize variables
        sess.run(init)

        # # If weights option is set, preload weights from existing models appropriately
        # if FLAGS.weights:
        #     logging.info("Loading weights from pretrained model - %s ", FLAGS.weights )
        #     ckpt = tf.train.get_checkpoint_state(FLAGS.weights)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #     else:
        #         logging.error("Weight file for pretrained model not found: %s", FLAGS.weights  )
        #         exit(-1)

        # Start queue runners
        # @TODO: LMDB-tf-reader Op needs to be plemented for us the use queue runners.
        # tf.train.start_queue_runners(sess=sess)

        # Tensorboard: Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        writer_train = tf.train.SummaryWriter(os.path.join(FLAGS.tf_summaries_dir, 'tb', 'train'), sess.graph)
        writer_val = tf.train.SummaryWriter(os.path.join(FLAGS.tf_summaries_dir, 'tb', 'val'), sess.graph)

#        # Start looping
#        for step in xrange(FLAGS.epoch * train_data_loader.total):
#            start_time = time.time()
#
#            data_loader_index = (step * batch_size_train) % (train_data_loader.total-batch_size_train) # hacky
#            data_batch_size = batch_size_train #@TODO
#            batch_x, batch_y = train_data_loader.next_batch(data_batch_size, data_loader_index)
#
#            _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_train'].items()))
#            duration = time.time() - start_time
#
#            if np.isnan(loss_value):
#                logging.error('Model diverged with loss = NaN')
#                exit(-1)
#
#            if step % 10 == 0:
#                #num_examples_per_step = FLAGS.batchSize * ngpus
#                #examples_per_sec = num_examples_per_step / duration
#                #sec_per_batch = duration / ngpus
#                logging.info("Training (step " + str(step) + "): loss = " + "{:.6f}".format(loss_value) + "): acc = " + "{:.6f}".format(acc)   )            
#
#            if step % 100 == 0:
#                summary_str = sess.run(summary_op, feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_train'].items()))
#                writer_train.add_summary(summary_str, step)
#                writer_train.flush()
#
#
#        # enforce clean exit
#        exit(0)     



        # Launch the graph
        for epoch in xrange(0,FLAGS.epoch):
            data_loader_index = 0
            t = 0
            logged_since_last_check = 0 # Amount of images logged since last logging

            # Start with an initial validation
            Validation(sess, x, y, val_data_loader, writer_val, batch_size_val, loss, accuracy, network, summary_op, 0, 0)

            while t < train_data_loader.total:
                step = t+epoch*train_data_loader.total # Usage of steps seems a TensorFlow convention
                data_batch_size = min(train_data_loader.total-data_loader_index, batch_size_train)

                batch_x, batch_y = train_data_loader.next_batch(data_batch_size, data_loader_index)

                # Update for next batch start index
                data_loader_index = data_loader_index + data_batch_size
                
                logged_since_last_check = logged_since_last_check + data_batch_size

                current_epoch = epoch + round(float(t)/train_data_loader.total, epoch_round)
                
                

                # Backward pass
                _, loss_val, acc_val, summary_str = sess.run([train_op, loss, accuracy, summary_op], feed_dict=dict({x: batch_x, y: batch_y}.items() + network['feed_dict_train'].items()))
                writer_train.add_summary(summary_str, step)
                if np.isnan(loss_val):
                    logging.error('Model diverged with loss_val = NaN')
                    exit(-1)

                # Start with a forward pass
                if (t == 0) or (logged_since_last_check >= logging_check_interval):
                    logged_since_last_check = 0
                    #TODO: report average loss and acc since last check? in the current way we only do it for the latest batch
                    logging.info("Training (epoch " + str(current_epoch) + "): loss = " + "{:.6f}".format(loss_val) + ", lr = " + str(FLAGS.learningRate)  + ", accuracy = " + "{:.5f}".format(acc_val) )            

                # Validation Pass
                if FLAGS.validation and current_epoch >= next_validation:
                    Validation(sess, x, y, val_data_loader, writer_val, batch_size_val, loss, accuracy, network, summary_op, step, current_epoch)

                    #To find next nearest epoch value that exactly divisible by FLAGS.interval:
                    next_validation = (round(float(current_epoch)/FLAGS.interval) + 1) * FLAGS.interval 
                    last_validation_epoch = current_epoch

                # Saving Snapshot
                if current_epoch >= next_snapshot_save:
                    filename_snapshot = os.path.join(FLAGS.save, snapshot_prefix + "_" + str(current_epoch) + "_Model.ckpt")
                    logging.info("Snapshotting to " + filename_snapshot)
                    saver.save(sess, filename_snapshot)
                    logging.info("Snapshot saved - " + filename_snapshot)

                    if FLAGS.tf_serving_export:
                        from tensorflow_serving.session_bundle import exporter
                        model_exporter = exporter.Exporter(saver)
                        # @TODO: The signature scores_tensor doesn't currently have a softmax layer.
                        signature = exporter.classification_signature(input_tensor=x, scores_tensor=network['model'])
                        model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
                        model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)

                    # To find next nearest epoch value that exactly divisible by FLAGS.snapshotInterval
                    next_snapshot_save = (round(float(current_epoch)/FLAGS.snapshotInterval) + 1) * FLAGS.snapshotInterval 
                    last_snapshot_save_epoch = current_epoch

                # The backward pass is completed, and we update the current epoch
                t = t + data_batch_size

        # We need to call sess.close() because we've used a with block
        sess.close()




if __name__ == '__main__':
    tf.app.run()     
                


