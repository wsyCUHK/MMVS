import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow.compat.v1 as tf
import tflearn

PPO_TRAINING_EPO = 5
ENTROPY_WEIGHT = 1
FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2


class Network:
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_00 = tflearn.embedding(inputs[:, 0:1, -1], input_dim=6, output_dim=2)
            split_ = tflearn.flatten(split_00)
            split_0 = tflearn.fully_connected(split_, FEATURE_NUM, activation='relu')

            split_11 = tflearn.embedding(inputs[:, 1:2, -1], input_dim=10, output_dim=2)
            split_1_ = tflearn.flatten(split_11)
            split_1 = tflearn.fully_connected(split_1_, FEATURE_NUM, activation='relu')

            split_2 = tflearn.lstm(inputs[:, 2:3, :], n_units=FEATURE_NUM)

            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            pi_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            pi = tflearn.fully_connected(pi_net, self.a_dim, activation='softmax')

        with tf.variable_scope('critic'):
            split_00 = tflearn.embedding(inputs[:, 0:1, -1], input_dim=6, output_dim=2)
            split_ = tflearn.flatten(split_00)
            split_0 = tflearn.fully_connected(split_, FEATURE_NUM, activation='relu')

            split_11 = tflearn.embedding(inputs[:, 1:2, -1], input_dim=10, output_dim=2)
            split_1_ = tflearn.flatten(split_11)
            split_1 = tflearn.fully_connected(split_1_, FEATURE_NUM, activation='relu')

            split_2 = tflearn.lstm(inputs[:, 2:3, :], n_units=FEATURE_NUM)

            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            value_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            value = tflearn.fully_connected(value_net, 1, activation='linear')
            return pi, value

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self._entropy_weight = ENTROPY_WEIGHT
        self.ppo_epo = PPO_TRAINING_EPO
        self.inner_value = 1
        self.outer_value = 0
        self.H_target = 0.1

        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)

        self.inner = tf.placeholder(tf.float32)
        self.mean = tf.placeholder(tf.float32)
        self.std = tf.placeholder(tf.float32)
        self.mean_out_value = tf.placeholder(tf.float32)
        self.std_out_value = tf.placeholder(tf.float32)

        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)


    def predict(self, input):
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0]

