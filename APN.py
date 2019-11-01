import tensorflow as tf
from MyFunc import myfc

"""
 featuremap = 5, 5, 256
"""
def APN_forward(featuremap, train, regularizer):
    shape = featuremap.get_shape().as_list()
    nodes = shape[1] * shape[2] * shape[3]
    reshaped = tf.compat.v1.reshape(featuremap, [shape[0], nodes])

    fc1 = myfc(reshaped, [nodes, 4096], regularizer, dropout=train, train=train)
    fc2 = myfc(fc1, [4096, 1024], regularizer, dropout=train, train=train)
    fc3 = myfc(fc2, [1024, 512], regularizer, dropout=train, train=train)
    logit = myfc(fc3, [512, 3], regularizer, dropout=False, relu=False, train=train)
    return logit





