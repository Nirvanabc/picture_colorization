import numpy as np
import tensorflow as tf
from play_with_image import *
from constants import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

is_training = tf.placeholder(tf.bool)

def conv_layer(kernel, in_chan, out_chan, tensor, strides):
    W_conv = weight_variable([kernel, kernel, in_chan,
                               out_chan])
    b_conv = bias_variable([out_chan])
    conv = tf.nn.conv2d(tensor, W_conv, strides = strides,
                        padding=padding)
    batch_norm = tf.layers.batch_normalization(
        inputs=conv,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=center,
        scale=scale,
        training = is_training)
    h = tf.nn.bias_add(batch_norm, b_conv)
    # h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
    return h

# target
y_input = tf.placeholder(tf.float32, \
                         shape = [None, height,
                                  width, out_chan], name='y')
y_ = tf.layers.batch_normalization(
    inputs=y_input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=center,
    scale=scale,
    training = is_training)

# input
x = tf.placeholder(tf.float32, \
                   shape = [None, height,
                            width, in_chan], name='x')
# x = tf.layers.batch_normalization(
#     inputs=x_input,
#     axis=-1,
#     momentum=0.9,
#     epsilon=0.001,
#     center=senter,
#     scale=scale,
#     training = is_training)
# 

### LOW_LAYER_FEATUES_NETWORK
# h*w*3 -> h/2*w/2*64 -> h/2*w/2*128
low_11 = conv_layer(kernel, in_chan, out_chan_11, x,
                  strides_2)
low_12 = conv_layer(kernel, in_chan_12, out_chan_12, low_11,
                  strides_1)


# h/2*w/2*128 -> h/4*w/4*128 -> h/4*w/4*256
low_21 = conv_layer(kernel, in_chan_21, in_chan_21, low_12,
                  strides_2)
low_22 = conv_layer(kernel, in_chan_22, out_chan_22, low_21,
                  strides_2)


### GLOBAL_FEATUES_NETWORK
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
# 
# total_shape = int(glob_22.shape[1]*glob_22.shape[2]*
#                                  glob_22.shape[3])
# x_fc_1 = tf.reshape(glob_22, [-1, total_shape])
# w_fc_1 = weight_variable([total_shape, out_chan_fc_1])
# b_fc_1 = bias_variable([out_chan_fc_1])
# glob_fc_1 = tf.nn.tf.nn.bias_add(tf.matmul(x_fc_1, w_fc_1), b_fc_1)
# 
# w_fc_2 = weight_variable([out_chan_fc_1, out_chan_fc_2])
# b_fc_2 = bias_variable([out_chan_fc_2])
# glob_fc_2 = tf.nn.bias_add(tf.matmul(glob_fc_1, w_fc_2), b_fc_2)

### middle level
mid_1 = conv_layer(kernel, in_chan_mid,
                 out_chan_mid, low_22, strides_1)
mid_2 = conv_layer(kernel, in_chan_mid,
                 out_chan_mid, mid_1, strides_1)


### colorization network
fusion_layer = conv_layer(kernel, out_chan_mid, out_chan_col_1,
                          mid_2, strides_1)


## A VERY WEAK PLACE -- float32 to uint8 !!!
# fusion_layer = tf.cast(fusion_layer, tf.int32)

def color_layer(layer, in_chan, out_chan):
    color_upsamp = tf.image.resize_nearest_neighbor(
        layer,
        [int(layer.shape[1])*2,
         int(layer.shape[2])*2])
    color_conv = conv_layer(
        kernel,
        in_chan,
        out_chan,
        color_upsamp,
        strides_1)
    return color_conv


color_layer_1 = color_layer(fusion_layer,
                            out_chan_col_1,
                            out_chan_col_1)

color_layer_2 = color_layer(color_layer_1,
                            out_chan_col_1,
                            out_chan_col_2)

color_layer_3 = color_layer(color_layer_2,
                            out_chan_col_2,
                            out_chan_col_3)


### READOUT LAYER
mul = tf.Variable(60.0)
add = tf.Variable(128.0)

W_read = weight_variable([kernel, kernel,
                          out_chan_col_3, out_chan])
b_read = bias_variable([out_chan])

conv = tf.nn.conv2d(color_layer_3, W_read, strides = strides_1,
                        padding=padding)

# to make real colors from scaled and sentered channels
h = conv * mul
y_conv = h + add

### now train and evaluate
correct_prediction = tf.norm(y_ - y_conv)
tf.summary.scalar('loss', correct_prediction)
# cross_entropy = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, \
#                                             logits=y_conv))
train_step = tf.train.AdadeltaOptimizer().minimize(
    correct_prediction)

merged = tf.summary.merge_all()

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(
        "output/train", sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for j in range(epochs):
        yield_batch = get_batch(batch_size)
        for i in range (iterations):
            batch = next(yield_batch)
            sess.run([train_step,
                      extra_update_ops],
                     feed_dict={
                         x: batch[0],
                         y_: batch[1],
                         is_training: True})
            
            if i % print_each == 0:
                acc, image_train_0, summary = sess.run(
                    [correct_prediction,
                     y_conv,
                     merged],
                    feed_dict = {x: batch[0],
                                 y_: batch[1],
                                 is_training: True})
                image_train = image_train_0[0]
                train_writer.add_summary(summary, i)
                predicted_image = np.concatenate((batch[0][0],
                                                  image_train),
                                                 axis=2)
                image_uint = sess.run(tf.cast(predicted_image,
                                              tf.uint8))
                rgb_image = cv2.cvtColor(image_uint,
                                         cv2.COLOR_LAB2BGR)
                cv2.imwrite("new_%d.jpeg" % i, rgb_image)
                print("step %d, acc %.2f epoch %d" % (i, acc, j))
            if i % save_each == 0:
                saver.save(sess, model_data, global_step=i)
