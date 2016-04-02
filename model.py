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
import os
import tensorflow as tf
import numpy as np

# Basic model parameters as external flags.
from parse import parse

flags = tf.app.flags
FLAGS = flags.FLAGS

# print(flags)
#
# defaults = [
#     (float, 'learning_rate', 0.0001, 'Initial learning rate')
# ]
#
# for typ, flag_name, default_value, docstring:
#     flags._define_helper(flag_name, default_value, docstring)
#     defaults[flag_name] = default_value

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('dropout_rate', .9, 'Probability of keeping nodes during dropout.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 1500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 800, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                                       'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_string('save_dir', 'checkpoints', 'Directory to save data from session.')
flags.DEFINE_string('summary_dir', 'logs', 'Directory to save data from session.')
flags.DEFINE_string('dataset', 'movies', 'Directory to put the training data.')
flags.DEFINE_boolean('debug', False, 'If true, use small dataset ')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')


def save_path(filename):
    return os.path.join(FLAGS.save_dir, filename)


CP_INFO = save_path('checkpoint')


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
    return {place_holder: feed for place_holder, feed
            in zip(place_holders, data_set.next_batch(FLAGS.batch_size))}


def restore_variables(sess):
    with open(CP_INFO) as cp_data:
        line = cp_data.readline()
        checkpoint = parse('model_checkpoint_path: "{}"\n', line).fixed[0]

    loader = tf.train.Saver()
    loader.restore(sess, save_path(checkpoint))
    print("Model restored.")


def run_training():
    """Train the autoencoder for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on data.
    data_sets = data.Data(datafile='debug.dat') if FLAGS.debug else data.Data()

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        placeholders = placeholder_inputs(FLAGS.batch_size, data_sets.dim)
        inputs_placeholder, labels_placeholder, mask_placeholder = placeholders

        # Build a Graph that computes predictions from the inference model.
        logits = ops.inference(inputs_placeholder,
                               FLAGS.dropout_rate,
                               data_sets.dim,
                               FLAGS.hidden1,
                               FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = ops.loss(logits, labels_placeholder, mask_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = ops.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct, eval_total = ops.evaluation(logits,
                                                  labels_placeholder,
                                                  mask_placeholder)

        # keep track of the epoch
        count = tf.Variable(0)
        increment = tf.count_up_to(count, FLAGS.num_epochs)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        if os.path.exists(CP_INFO):
            restore_variables(sess)

            # redo epoch that we last quit in the middle of
            sess.run(count.assign_sub(1))
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir,
                                                graph_def=sess.graph_def)

        def do_eval(data_set):
            """Runs one evaluation against the full epoch of data.
              :param data_set: The data_set which we will use to retrieve batches
          """
            # And run one epoch of eval.
            steps_per_epoch = data_set.num_examples // FLAGS.batch_size
            counts = np.zeros(2)
            for _ in xrange(steps_per_epoch):
                feed_dict = fill_feed_dict(data_set, placeholders)
                run = sess.run([eval_correct, eval_total], feed_dict=feed_dict)
                counts += run
            correct, total = counts
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (int(total), int(correct), correct / total))

        # And then after everything is built, start the training loop.
        steps_per_epoch = data_sets.train.num_examples // FLAGS.batch_size
        start_time = time.time()

        # TODO: make epoch a saved variable so that training picks up where it left off
        for _ in xrange(FLAGS.num_epochs):
            epoch = sess.run(increment)
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

                if step == 0:
                    # Write the summaries and print an overview fairly often.
                    duration = time.time() - start_time

                    # Print status to stdout.
                    print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, duration))

                    # Update the events file.
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, epoch)

                    # create the save directory if not already there
                    if not os.path.isdir(FLAGS.save_dir):
                        os.mkdir(FLAGS.save_dir)
                    save_dir = os.path.join(FLAGS.save_dir, 'model.ckpt')
                    saver.save(sess, save_dir, global_step=step)

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


def predict(instance, dat):
    with tf.Graph().as_default():
        # Build a Graph that computes predictions from the inference model.
        logits = ops.inference(tf.constant(instance),
                               1,  # no dropout
                               dat.dim,
                               FLAGS.hidden1,
                               FLAGS.hidden2)

        sess = tf.Session()
        # Restore variables from disk.
        restore_variables(sess)
        predictions = sess.run(logits)
        return data.unnormalize(predictions)


def main(_):
    run_training()


if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        print('\nGoodbye.')
        exit(0)
