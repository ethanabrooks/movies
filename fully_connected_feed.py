# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import data
import ops
import tensorflow as tf
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 2500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 2500, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                                       'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_string('save_dir', 'save', 'Directory to save data from session.')
flags.DEFINE_string('dataset', 'movies', 'Directory to put the training data.')
flags.DEFINE_boolean('debug', False, 'If true, use small dataset ')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')


def placeholder_inputs(batch_size, data_dim):
    """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    :param data_dim: the dimension of our input data and output data
  """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.

    shape = (batch_size, data_dim)
    images_placeholder = tf.placeholder(tf.float32, shape=shape)
    labels_placeholder = tf.placeholder(tf.float32, shape=shape)
    mask_placeholder = tf.placeholder(tf.bool, shape=shape)

    return images_placeholder, labels_placeholder, mask_placeholder


def fill_feed_dict(data_set, place_holders):
    """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

    :param data_set: The set of movies and ratings, from data.DataSets()
    :param place_holders: The three placeholders, from placeholder_inputs(),
    including the inputs placeholder, the labels placeholder
    -- identical to inputs placeholder except for the dropout values --
    and the mask placeholder, from placeholder_inputs(),
    which will receive a tensor of booleans for masking missing data values

    :returns feed_dict: The feed dictionary mapping from placeholders to values.
  """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, labels_feed, mask_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {place_holders[i]: feed for i, feed
                 in enumerate((images_feed, labels_feed, mask_feed))}

    return feed_dict


def run_training():
    """Train the autoencoder for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on data.
    data_sets = data.DataSets(datafile='debug.dat') if FLAGS.debug else data.DataSets()

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        placeholders = placeholder_inputs(FLAGS.batch_size, data_sets.dim)
        images_placeholder, labels_placeholder, mask_placeholder = placeholders

        # Build a Graph that computes predictions from the inference model.
        logits = ops.inference(images_placeholder,
                               data_sets.dim,
                               FLAGS.hidden1,
                               FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = ops.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = ops.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct, eval_total = ops.evaluation(logits,
                                                  labels_placeholder,
                                                  mask_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # sess = tf.InteractiveSession()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.save_dir,
                                                graph_def=sess.graph_def)

        # And then after everything is built, start the training loop.
        steps_per_epoch = data_sets.train.num_examples // FLAGS.batch_size + 1


        def do_eval(data_set):
            """Runs one evaluation against the full epoch of data.
              :param data_set: The data_set which we will use to retrieve batches
          """
            # And run one epoch of eval.
            counts = np.zeros(2)
            for _ in xrange(steps_per_epoch):
                feed_dict = fill_feed_dict(data_set, placeholders)
                run = sess.run([eval_correct, eval_total], feed_dict=feed_dict)
                counts += run
            correct, total = counts
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (int(total), int(correct), correct / total))

        bar = data.progress_bar('Training')
        start_time = time.time()
        for epoch in xrange(FLAGS.num_epochs):
            for step in xrange(steps_per_epoch):
                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                feed_dict = fill_feed_dict(data_sets.train, placeholders)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                # Write the summaries and print an overview fairly often.
                bar.next()
            duration = time.time() - start_time

            # Print status to stdout.
            print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, duration))

            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, epoch)
            saver.save(sess, FLAGS.save_dir, global_step=step)

            if epoch % 10 == 0 or (epoch + 1) == FLAGS.num_epochs:
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(data_sets.test)
        bar.finish()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
