import tensorflow as tf
from MyFunc import myconv, mypool, myfc

IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 8

CONV1_DEEP = 96
CONV1_SIZE = 11

CONV2_DEEP = 256
CONV2_SIZE = 5

CONV3_DEEP = 384
CONV3_SIZE = 3

CONV4_DEEP = 384
CONV4_SIZE = 3

CONV5_DEEP = 256
CONV5_SIZE = 3

FC1_SIZE = 4096
FC2_SIZE = 4096
keep_prob = 0.5

# Using AlexNet Model
def classify_forward(input_tensor, train, regularizer):
    conv1 = myconv(input_tensor, [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], [1, 4, 4, 1], "VALID")
    pool1 = mypool(conv1)
    conv2 = myconv(pool1, [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
    pool2 = mypool(conv2)
    conv3 = myconv(pool2, [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP])
    conv4 = myconv(conv3, [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP])
    conv5 = myconv(conv4, [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP])
    pool3 = mypool(conv5)

    featuremap = pool3

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.compat.v1.reshape(pool3, [pool_shape[0], nodes])

    fc1 = myfc(reshaped, [nodes, FC1_SIZE], regularizer, dropout=train, train=train)
    fc2 = myfc(fc1, [FC1_SIZE, FC2_SIZE], regularizer, dropout=train, train=train)
    logit = myfc(fc2, [FC2_SIZE, NUM_LABELS], regularizer, dropout=False, relu=False, train=train)

    return featuremap, logit