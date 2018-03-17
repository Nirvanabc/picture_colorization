import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from play_with_image import *
from constants import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, \
                   shape = [None, height,
                            width, in_chan_11], name='x')

# target
y_ = tf.placeholder(tf.uint8, \
                    shape = [None, height,
                             width, in_chan_11], name='y')

# x_tensor = tf.reshape(x, [-1, height, width, in_chan_11])


def conv_layer(kernel, in_chan, out_chan, tensor, strides):
    W_conv = weight_variable([kernel, kernel, in_chan,
                               out_chan])
    b_conv = bias_variable([out_chan])
    conv = tf.nn.conv2d(tensor, W_conv, strides = strides,
                        padding=padding)
    h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
    return h


### LOW_LAYER_FEATUES_NETWORK
# h*w*3 -> h/2*w/2*64 -> h/2*w/2*128
low_11 = conv_layer(kernel, in_chan_11, out_chan_11, x,
                  strides_2)
low_12 = conv_layer(kernel, in_chan_12, out_chan_12, low_11,
                  strides_1)

# h/2*w/2*128 -> h/4*w/4*128 -> h/4*w/4*256
low_21 = conv_layer(kernel, in_chan_21, in_chan_21, low_12,
                  strides_2)
low_22 = conv_layer(kernel, in_chan_22, out_chan_22, low_21,
                  strides_2)


### GLOBALOW_FEATUES_NETWORK
# h/4*w/4*256 -> h/8*w/8*256 -> h/8*w/8*256
# fc_512 -> fc_256
# glob_11 = conv_layer(kernel, chan, chan, low_22,
#                   strides_2)
# 
# glob_12 = conv_layer(kernel, chan, chan, glob_11,
#                   strides_1)
# 
# 
# glob_21 = conv_layer(kernel, chan, chan, glob_12,
#                   strides_2)
# 
# glob_22 = conv_layer(kernel, chan, chan, glob_21,
#                   strides_1)
# 
# total_shape = int(glob_22.shape[1]*glob_22.shape[2]*
#                                  glob_22.shape[3])
# x_fc_1 = tf.reshape(glob_22, [-1, total_shape])
# w_fc_1 = weight_variable([totalow_shape, out_chan_fc_1])
# b_fc_1 = bias_variable([out_chan_fc_1])
# glob_fc_1 = tf.nn.bias_add(tf.matmul(x_fc_1, w_fc_1), b_fc_1)
# 
# w_fc_2 = weight_variable([out_chan_fc_1, out_chan_fc_2])
# b_fc_2 = bias_variable([out_chan_fc_2])
# glob_fc_2 = tf.nn.bias_add(tf.matmul(glob_fc_1, w_fc_2), b_fc_2)
# 

### middle level
mid_1 = conv_layer(kernel, in_chan_mid,
                 out_chan_mid, low_22, strides_1)
mid_2 = conv_layer(kernel, in_chan_mid,
                 out_chan_mid, mid_1, strides_1)


### colorization network
fusion_layer = conv_layer(kernel, out_chan_mid, out_chan_col_1,
                          mid_2, strides_1)
# glob_fc_2 = tf.reshape(glob_fc_2, [
# fusion_layer = tf.concat([glob_fc_2, mid_2], 3)

## A VERY WEAK PLACE -- float32 to uint8 !!!
# fusion_layer = tf.cast(fusion_layer, tf.int32)


color_upsamp_1 = tf.image.resize_nearest_neighbor(fusion_layer,
                             [int(fusion_layer.shape[1])*2,
                              int(fusion_layer.shape[2])*2])
color_conv_1 = conv_layer(kernel, out_chan_col_1, out_chan_col_1,
                         color_upsamp_1, strides_1)

color_upsamp_2 = tf.image.resize_nearest_neighbor(color_conv_1,
                             [int(color_conv_1.shape[1])*2,
                              int(color_conv_1.shape[2])*2])
color_conv_2 = conv_layer(kernel, out_chan_col_1, out_chan_col_2,
                         color_upsamp_2, strides_1)

color_upsamp_3 = tf.image.resize_nearest_neighbor(color_conv_2,
                             [int(color_conv_2.shape[1])*2,
                              int(color_conv_2.shape[2])*2])
color_conv_3 = conv_layer(kernel, out_chan_col_2, out_chan_col_3,
                         color_upsamp_3, strides_1)

# yield_batch = convert_to_grayscale(batch_size)


### READOUT LAYER
W_read = weight_variable([out_chan_col_3, class_num])
b_read = bias_variable([class_num])

y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)


### now train and evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, \
                                            logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), \
                              tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, \
                                  tf.float32), name = 'accuracy')

yield_batch = convert_to_grayscale(batch_size)
test_batch = next(yield_batch)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        batch = next(yield_batch)
        sess.run(train_step, feed_dict={x: batch[0],
                                        y_: batch[1]})
        acc = sess.run(accuracy, feed_dict={x:test_batch[0],
                                            y_: test_batch[1]})
        print("step %d, acc %.2f" % (i, acc))
        
