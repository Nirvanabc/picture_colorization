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
y_conv = conv_layer(kernel, out_chan_col_3, out_chan,
                    color_layer_3, strides_1)

### now train and evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, \
                                            logits=y_conv))
train_step = tf.train.AdadeltaOptimizer().minimize(
    cross_entropy)
correct_prediction = tf.norm(y_ - y_conv)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

def map_func(x):
    tmp = x
    for raw in tmp:
        for col in raw:
            for i,num in enumerate(col):
                col[i] *= 70
                col[i] += 120
    return np.array(tmp)


yield_batch = convert_to_grayscale(batch_size)

def give_next(yield_obg, batch_size):
    try:
        batch = next(yield_obg)
    except StopIteration:
        yield_obj = convert_to_grayscale(batch_size)
        batch = next(yield_obj)
    finally: return batch


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        # batch = give_next(yield_batch, batch_size)
        batch = next(yield_batch)
        sess.run(
            [extra_update_ops,
             train_step],
            feed_dict={
                x: batch[0],
                y_input: batch[1],
                is_training: True})
        
        if i % print_each == 0:
            _, acc, image_train_0 = sess.run( 
                [extra_update_ops,
                 correct_prediction,
                 y_conv],
                feed_dict= {x: batch[0],
                            y_input: batch[1],
                            is_training: False})
            
            image_train = image_train_0[0]
            # image_train = map_func(image_train_0[0])
            predicted_image = np.concatenate((batch[0][0],
                                              image_train),
                                             axis=2)
            image_uint = sess.run(tf.cast(predicted_image,
                                          tf.uint8))
            rgb_image = cv2.cvtColor(image_uint,
                                     cv2.COLOR_LAB2BGR)
            cv2.imwrite("new_%d.jpeg" % i, rgb_image)
            print("step %d, acc %.2f" % (i, acc))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.norm(y_), feed_dict = {y_input: batch[1], is_training: True})
