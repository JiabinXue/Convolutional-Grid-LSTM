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
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy as np
class CNNGridLSTMCell():
    def __init__(self,
                 filter_size,
                 filter_stride,
                 pooling_size,
                 pooling_stride,
                 share_time_frequency_weights=False,
                 feature_size=None,
                 frequency_skip=None,
                 num_frequency_blocks=None,
                 start_freqindex_list=None,
                 end_freqindex_list=None,
                 use_bn = False,
                 is_training=False
                 ):
        self._filter_size = filter_size
        self._filter_stride = filter_stride
        self._pooling_size = pooling_size
        self._pooling_stride = pooling_stride
        self._share_time_frequency_weights = share_time_frequency_weights
        self._feature_size = feature_size
        self._frequency_skip = frequency_skip
        self._start_freqindex_list = start_freqindex_list
        self._end_freqindex_list = end_freqindex_list
        self._num_frequency_blocks = num_frequency_blocks
        self._total_blocks = 0
        self.init = tf.glorot_uniform_initializer()
        self.use_bn = use_bn
        self.is_training = is_training

    def call(self, inputs):
        shape = tf.shape(inputs)  #300 * 999 * 80
        freq_inputs = self._make_tf_features(inputs) #8 * 294 * 300 * 40 * 10 * 3
        out_lst = []
        self.o = []
        #self.o = freq_inputs
        for block in range(len(freq_inputs)):
            out_lst_current = self._compute(array_ops.concat(freq_inputs[block], 0), block) # (294 * 300) * 256

            out_lst.append(out_lst_current) #300 * 294 * 256
            self.o.append(out_lst_current)
        m_out = array_ops.concat(out_lst, 2) # 300 * 294 * (256 * 8)
        return m_out

    def _compute(self, freq_inputs, block): # (294 * 300) * 40 * 10 * 3
        if self._share_time_frequency_weights == True:
            with tf.name_scope("CNN"):
                with tf.name_scope("conv-maxpool"):
                    #W = tf.Variable(tf.truncated_normal(self._filter_size, stddev=0.05), name="W")
                    #b = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
                    W = tf.Variable(self.init(self._filter_size), name="W")
                    b = tf.Variable(self.init([32]), name="b")
                    conv = tf.nn.conv2d(freq_inputs, W, strides=self._filter_stride, padding="SAME", name="conv") #(294 * 300) * 40 * 10 * 64
                    if self.use_bn:
                        conv = tf.layers.batch_normalization(conv, self.is_training)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                            h,
                            ksize=self._pooling_size,
                            strides=self._pooling_stride,
                            padding='SAME',
                            name="pool")   #(294 * 300) * 20 * 5 * 64
            with tf.name_scope("DnnLayer"):
                shape = tf.shape(pooled)
                pooled_flat = tf.reshape(pooled, [-1, shape[1], int(self._filter_size[-1] * (self._end_freqindex_list[0] - self._start_freqindex_list[0]) / 2)]) # (294 * 300) * (20 * 5 * 64)
                #w = tf.Variable(tf.truncated_normal([8 * 1 * 32, 128], stddev=0.01))
                #b = tf.Variable(tf.zeros([128]))
                #w = tf.Variable(self.init([8 * 1 * 32, 128]))
                #b = tf.Variable(self.init([128]))
                #logist = tf.matmul(pooled_flat, w) + b
                #logist = tf.reshape(logist, [shape[0], 128])
                logist = pooled_flat
        else:
            with tf.name_scope("CNN%d" %(block)):
                with tf.name_scope("conv-maxpool"):
                    #W = tf.Variable(tf.truncated_normal(self._filter_size, stddev=0.05), name="W")
                    #b = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
                    W = tf.Variable(self.init(self._filter_size), name="W")
                    b = tf.Variable(self.init([32]), name="b")
                    conv = tf.nn.conv2d(freq_inputs, W, strides=self._filter_stride, padding="SAME", name="conv") #(294 * 300) * 40 * 10 * 64
                    #conv = tf.layers.BatchNormalization()(conv)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                            h,
                            ksize=self._pooling_size,
                            strides=self._pooling_stride,
                            padding='SAME',
                            name="pool")   #(294 * 300) * 20 * 5 * 64

            with tf.name_scope("DnnLayer"):
                shape = tf.shape(pooled)
                pooled_flat = tf.reshape(pooled, [-1, shape[1], int(self._filter_size[-1] * (self._end_freqindex_list[0] - self._start_freqindex_list[0]) / 2)]) # (294 * 300) * (20 * 5 * 64)
                #w = tf.Variable(tf.truncated_normal([1 * 8 * 32, 128], stddev=0.01))
                #b = tf.Variable(tf.zeros([128]))
                #w = tf.Variable(self.init([1 * 8 * 32, 128]))
                #b = tf.Variable(self.init([128]))
                #logist = tf.matmul(pooled_flat, w) + b
                #logist = tf.reshape(logist, [shape[0], 128])
                logist = pooled_flat
            pass
        return logist

    def _make_tf_features(self, input_feat, slice_offset=0):

        input_size = input_feat.get_shape().with_rank(3)[-1].value
        #input_size = 1701
        #print(input_size)
        shape = tf.shape(input_feat)
        tf.logging.info(shape)

        #input_feat = tf.reshape(input_feat, [shape[0], -1, 3, 80])
        #input_feat = tf.transpose(input_feat, [0, 1, 3, 2])
        input_feat = tf.expand_dims(input_feat, -1)
        if slice_offset > 0:
            # Padding to the end
            inputs = array_ops.pad(input_feat,
                                   array_ops.constant(
                                       [0, 0, 0, slice_offset],
                                       shape=[2, 2],
                                       dtype=dtypes.int32), "CONSTANT")
        elif slice_offset < 0:
            # Padding to the front
            inputs = array_ops.pad(input_feat,
                                   array_ops.constant(
                                       [0, 0, -slice_offset, 0],
                                       shape=[2, 2],
                                       dtype=dtypes.int32), "CONSTANT")
            slice_offset = 0
        else:
            inputs = input_feat
        freq_inputs = []
        for b in range(len(self._start_freqindex_list)):
            start_index = self._start_freqindex_list[b]
            end_index = self._end_freqindex_list[b]
            cur_size = end_index - start_index
            block_feats = int(
                (input_size - self._feature_size) / (self._frequency_skip)) + 1
            block_inputs = []
            #for f in range(block_feats):
            cur_input = array_ops.slice(
                inputs,
                [0, 0, start_index, 0],
                [-1, -1, cur_size, -1])
            #print(tf.shape(cur_input).eval(feed_dict={input: np.zeros([32, 1701, 240])}, session=tf.Session()))
            block_inputs.append(cur_input)
            freq_inputs.append(block_inputs)
        return freq_inputs


if __name__ == '__main__':
    Grid = CNNGridLSTMCell(filter_size=[1, 3, 1, 32],
                           filter_stride=[1, 1, 1, 1],
                           pooling_size=[1, 1, 2, 1],
                           pooling_stride=[1, 1, 2, 1],
                           share_time_frequency_weights=False,
                           feature_size=1,
                           frequency_skip=1,
                           num_frequency_blocks=None,
                           start_freqindex_list=[i for i in range(0,240-8,8)],
                           end_freqindex_list=[i for i in range(16,240+8,8)])
    input = tf.placeholder(shape=[None, None, 240], dtype=tf.float32)
    res = Grid.call(input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(Grid.o, feed_dict={input: np.zeros([32, 1701, 240])})
        #print(np.array(a).shape)
        for i in a:
            print(np.array(i).shape)
    pass
