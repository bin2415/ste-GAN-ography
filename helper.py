import tensorflow as tf
import utils

def fc_layer(x, shape, name, lasted):
    num_inputs, num_outputs = shape
    W = utils.weight_variable(shape, 1.0, name + "/W")
    b = utils.bias_variable([num_outputs], 0.0, name + "/b")
    if lasted:
        return tf.nn.tanh(tf.matmul(x, W) + b)
    else:
        return tf.nn.relu(tf.matmul(x, W) + b)
