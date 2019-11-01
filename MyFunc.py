import tensorflow as tf


def myconv(input_tensor, shape, strides=[1, 1, 1, 1], padding='SAME'):

    conv_weights = tf.compat.v1.get_variable("weight", shape, initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.compat.v1.float32))
    conv_biases = tf.compat.v1.get_variable("bias", shape[3], initializer=tf.compat.v1.constant_initializer(0.0))
    conv = tf.compat.v1.nn.conv2d(input_tensor, conv_weights, strides=strides, padding=padding)
    relu = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, conv_biases))
    return relu


def mypool(input_tensor, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'):
    pool = tf.compat.v1.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
    return pool


def myfc(input_tensor, shape, regularizer, train, relu=True, dropout=True):

        fc_weights = tf.compat.v1.get_variable("weight", shape, initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        if train is True:
            tf.compat.v1.add_to_collection('losses', regularizer(fc_weights))

        fc_biases = tf.compat.v1.get_variable("bias", shape[1], initializer=tf.compat.v1.constant_initializer(0.1))

        fc = tf.compat.v1.matmul(input_tensor, fc_weights) + fc_biases
        if relu:
            fc = tf.compat.v1.nn.relu(fc)
        if dropout is True:
            fc = tf.compat.v1.nn.dropout(fc, 0.5)
        return fc