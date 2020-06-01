from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow.python.ops import math_ops
from absl import logging

from mnvtf.model_classes import ConvModel
from mnvtf.resnet_model import Model
from mnvtf.bilinearloss import bilinear_loss
from mnvtf.constants import CONF_MAT, ALPHA, VF_NET, RESNET


# Global variables
pen_mat = CONF_MAT

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

class EstimatorFns:
    def __init__(self, nclasses=6, cnn_model=VF_NET):
        self._nclasses = nclasses
        self._model = cnn_model

    def est_model_fn(self, features, labels, mode, params):

        logging.info(".............................................................")
        logging.info("Parameters for function est_model_fn...") #Oscar erase
        logging.info("Features: {}".format(features)) #Oscar erase
        logging.info("Labels: {}".format(labels)) #Oscar
        logging.info("Mode: {}".format(mode)) #Oscar erase
        logging.info("Params: {}".format(params)) #Oscar erase
        logging.info(".............................................................")

        # Choose model for training
        if self._model == RESNET:
            logging.info("Resnet Model selected...") #Oscar erase
            resnet_size = 50
            model = Model(
                resnet_size=resnet_size,
                bottleneck=True,
                num_classes=self._nclasses,
                num_filters=64,
                kernel_size=3, #7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                block_sizes=_get_block_sizes(resnet_size),
                block_strides=[1, 2, 2, 2],
#                resnet_version=resnet_version,
                data_format='channels_first'
#                dtype=dtype
                )
            logits = model(features['concat'], training=True)

        elif self._model == VF_NET:
            logging.info(".............................................................") #Oscar erase
            logging.info("Vertex Finding model selected...") #Oscar erase
            logging.info("Entering to ConvModel ...") #Oscar erase
            logging.info(".............................................................")

            model = ConvModel(self._nclasses)
            logits = model(features['x_img'],
                           features['u_img'],
                           features['v_img'])

            logging.info(".............................................................")
            logging.info("logits are: {}".format(logits)) #Oscar erase
            logging.info(".............................................................")

        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
                'eventids': features['eventids']
            }

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                }
            )

        else:

#          loss = bilinear_loss(labels, logits, pen_mat, alpha=ALPHA)
          loss = tf.compat.v1.losses.softmax_cross_entropy(
              onehot_labels=labels, logits=logits)
          accuracy = tf.compat.v1.metrics.accuracy(
              labels=tf.argmax(labels, axis=1),
              predictions=tf.argmax(logits, axis=1)
          )

        if mode == tf.estimator.ModeKeys.TRAIN:

            tf.compat.v1.summary.scalar('accuracy', accuracy[1])
            # If we are running multi-GPU, we need to wrap the optimizer!
            if self._model == VF_NET:
                optimizer = tf.compat.v1.train.MomentumOptimizer(
                    learning_rate=0.0025, momentum=0.9, use_nesterov=True
                ) #GradientDescentOptimizer(
            elif self._model == RESNET:
                optimizer = tf.compat.v1.train.AdamOptimizer() #epsilon=1e-08

            # Name tensors to be logged with LoggingTensorHook (??)
            tf.identity(loss, 'cross_entropy_loss')

            # Save accuracy scalar to Tensorboard output (loss auto-logged)
            #tf.compat.v1.summary.scalar('train_accuracy', accuracy)
            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss" : loss, "accuracy" : accuracy[1]},
                every_n_iter=500
            )
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(
                    loss,  tf.compat.v1.train.get_or_create_global_step()
                ),
                training_hooks = [logging_hook]
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            # we get loss 'for free' as an eval_metric
            save_acc_hook = tf.train.SummarySaverHook(
                save_steps=1500,
                output_dir='/data/minerva/omorenop/tensorflow/models/tests',
                summary_op=tf.compat.v1.summary.scalar('eval_acc', accuracy[1])
            )
            save_loss_hook = tf.train.SummarySaverHook(
                save_steps=1500,
                output_dir='/data/minerva/omorenop/tensorflow/models/tests',
                summary_op=tf.compat.v1.summary.scalar('eval_loss', loss)
            )
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={'accuracy':accuracy},
                evaluation_hooks=[save_acc_hook, save_loss_hook]
            )

        return None


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    :return: ServingInputReciever
    """
    reciever_tensors = {
        # The size of input image is flexible. Oscar modified this
        'eventids' : tf.placeholder(tf.int64, [None]),
        'x_img': tf.placeholder(tf.float32, [None, 2, 127, 104]),
        'v_img': tf.placeholder(tf.float32, [None, 2, 127, 52]),
        'u_img': tf.placeholder(tf.float32, [None, 2, 127, 52]),
    }
    #'x_img': tf.placeholder(tf.float32, [None, 2, 127, 104]), #127, 24
    #'v_img': tf.placeholder(tf.float32, [None, 2, 127, 52]), #127, 12
    #'u_img': tf.placeholder(tf.float32, [None, 2, 127, 52]), #127, 12

    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=reciever_tensors,
        features=reciever_tensors
        )


def loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is
    smaller.
    Both evaluation results should have the values for MetricKeys.LOSS, which
    are used for comparison.
    Args:
        best_eval_result: best eval metrics.
        current_eval_result: current eval metrics.
    Returns:
        True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
        ValueError: If input eval result is None or no loss is available.
    """
    default_key = metric_keys.MetricKeys.LOSS
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]
