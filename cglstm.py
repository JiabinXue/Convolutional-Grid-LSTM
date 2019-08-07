
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import math
import tensorflow as tf
from moe import create_moe
from class_prior import get_class_prior
from cnn_cell import CNNGridLSTMCell


def cglstm(nnet_input, input_dim, sequence_length, f_l, f_s, use_bn, is_training):
    seed = None
    reg_loss =[]

    nnet_input_shape = tf.shape(nnet_input)
    batch_size = nnet_input_shape[0]
    with tf.variable_scope("cnn"):
        Grid = CNNGridLSTMCell(filter_size=[1, 3, 1, 32],
                                      filter_stride=[1, 1, 1, 1],
                                      pooling_size=[1, 1, 2, 1],
                                      pooling_stride=[1, 1, 2, 1],
                                      share_time_frequency_weights=False,
                                      feature_size=1,
                                      frequency_skip=1,
                                      num_frequency_blocks=None,
                                      start_freqindex_list=[i for i in range(0, input_dim - f_l + 1, f_s)],
                                      end_freqindex_list=[i for i in range(f_l, input_dim + 1, f_s)],
                                      use_bn = use_bn,
                                      is_training = is_training)
    nnet_input = Grid.call(nnet_input)
    nnet_input_shape = tf.shape(nnet_input)
    # Building GLSTMs
    with tf.variable_scope("grnn"):
        #m_cell = tf.contrib.rnn.DropoutWrapper(
        m_cell = tf.contrib.rnn.GridLSTMCell(num_units=128, 
                                    use_peepholes=False, 
                                    num_unit_shards=1, 
                                    forget_bias=5.0, 
                                    feature_size=f_l / 2 * 32,
                                    frequency_skip=f_l / 2 * 32, 
                                    num_frequency_blocks=[int(input_dim - f_l) / f_s + 1],
                                    share_time_frequency_weights=True)


    grid_initial_states = m_cell.zero_state(
                  batch_size=batch_size,
                  dtype=tf.float32,
              )
    grid_output, _ = \
            tf.nn.dynamic_rnn(
                cell=m_cell,
                inputs=nnet_input,
                sequence_length=sequence_length,
                initial_state=grid_initial_states,
                dtype=tf.float32,
                scope="grid"
            ) 
    return grid_output
