from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from absl import logging
from itertools import cycle
from mnvtf.constants import PREDICT, VF_NET, RESNET


def _make_generator_fn(files, batch_size, nclasses=195,
                       target_field='vtx_data/planecodes',
                       mode=None, max_evts=None):
    """
    'files' must be an iterable element containing paths for all files.
    'max_evts' is the maximum number of events to be taken from files.
    Make a generator function that we can query for batches. We want to train
    and evaluate over all elements in our file lists and start over for several
    epochs. In 'predict' mode we want to stop reading when we reach the
    end of last file in the list.
    """
    from mnvtf.hdf5_readers import SimpleCategorialHDF5Reader as HDF5Reader
    total_events = 0
    if mode == PREDICT:
        for ifile in files:
            reader = HDF5Reader(ifile)
            total_events += reader.openf()
            reader.closef()
        total_events = int(total_events / batch_size) * batch_size
        max_evts = max_evts if max_evts < total_events else total_events
        show_interval = round(max_evts / 10)

    def example_generator_fn():
        event_count = 0
        end_file_list = False
        for ifile in cycle(files):
            if not end_file_list:
                logging.debug("Reading {}".format(ifile))
                start_idx, stop_idx = 0, batch_size
                reader = HDF5Reader(ifile, target_field=target_field,
                                    nlabels=nclasses)
                nevents = reader.openf()

                while stop_idx <= nevents:
                    # use 'max_evts' instead of PREDICT since we might want to
                    # limit the number of events for training
                    if (max_evts is not None and
                        event_count + batch_size > max_evts):
                        end_file_list = True
                        break

                    if mode == PREDICT and event_count % show_interval == 0:
                        msg='Predicting [{}/{}]'.format(
                            int((event_count + show_interval) / batch_size),
                            int(max_evts / batch_size))
                        logging.info(msg)

                    yield reader.get_samples(start_idx, stop_idx)

                    event_count += batch_size
                    start_idx += batch_size
                    stop_idx += batch_size

                reader.closef()

            if ifile==files[-1]:
                if mode == PREDICT:
                    return
                else:
                    event_count = 0
                    end_file_list = False

        return

    return example_generator_fn


def make_dset(files, batch_size, nclasses=20,
              target_field='vtx_data/planecodes',
              max_evts=None, shuffle=False, mode=None): #hadro_data/n_hadmultmeas
    # make a generator function - read from HDF5
    dgen = _make_generator_fn(files, batch_size, nclasses,
                              target_field, mode, max_evts)

    # make a Dataset from a generator
    x_shape  = [None, 2, 127, 104] # Oscar modified this part
    uv_shape = [None, 2, 127, 52]  #
#    x_shape  = [None, 2, 127, 24] # Oscar modified this part for the reduced part
#    uv_shape = [None, 2, 127, 12]  #

    labels_shape = [None, nclasses]
    evtids_shape = [None]
    # TF doesn't support uint{32,64}, but leading bit should be zero for us
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.int64, tf.int32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(evtids_shape),
         tf.TensorShape(labels_shape),
         tf.TensorShape(x_shape),
         tf.TensorShape(uv_shape),
         tf.TensorShape(uv_shape))
    )
    # we are grabbing an entire "batch", so don't call `batch()`, etc.
    # also, note, there are issues with doing more than one epoch for
    # `from_generator` - so do just one epoch at a time for now.
    ds = ds.prefetch(10)
    if shuffle:
        ds = ds.shuffle(10)

    return ds


def make_iterators(files, batch_size, nclasses=195,
                   target_field='vtx_data/planecodes', max_evts=None,
                   shuffle=False, mode=None, cnn_model=VF_NET):
    '''
    estimators require an input fn returning `(features, labels)` pairs, where
    `features` is a dictionary of features.
    '''
    ds = make_dset(files, batch_size, nclasses=nclasses,
                   target_field=target_field, max_evts=max_evts,
                   shuffle=shuffle, mode=mode)

    # one_shot_iterators do not have initializers
    itrtr = tf.compat.v1.data.make_one_shot_iterator(ds)
    eventids, labels, x_img, u_img, v_img = itrtr.get_next()
    features = {}

    if cnn_model == VF_NET:
        features['x_img'] = x_img
        features['u_img'] = u_img
        features['v_img'] = v_img
    elif cnn_model == RESNET:
        u_img = tf.keras.backend.repeat_elements(u_img, 2, axis=3)
        v_img = tf.keras.backend.repeat_elements(v_img, 2, axis=3)
        concat = tf.concat([x_img, u_img, v_img], axis=1)
        features['concat'] = concat
    else:
        msg ="'{}' is not a valid model. You should use either '{}' or" + \
            "'{}'.".format(cnn_model, VF_NET, RESNET)
        ValueError (msg)

    features['eventids'] = eventids

    return features, labels
