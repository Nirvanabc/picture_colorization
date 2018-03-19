import tensorflow as tf
import numpy as np
import cv2
from constants import *


## FIXME!
# 1) It should give any size any time, not set in
# the beginning
# 2) It should return array of real sizes to print real image
# if I want
def convert_to_grayscale(batch_size):
    '''
    use next(batch) to obtain image list
    '''
    
    batch = []
    count = 0
    for i in range(1, max_length - max_length % batch_size + 1):
        index = '0'*(4 - len(str(i))) + str(i)
        image = cv2.imread("airplane/image_" +
                           index +
                           ".jpg")
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        square_image = tf.image.resize_images(image_yuv,
                                              [height, width])
        batch.append(square_image)
        if i % batch_size == 0:
            with tf.Session() as sess:
                batch = sess.run(batch)
            batch = np.array(batch)
            yield [batch[:, :, :, :1], batch[:, :, :, 1:3]]
            batch = []
            count = 0

# It works well: resizing image to 600x600 and back,
# I can't see the difference


# image = cv2.imread("1.JPEG")
# # box_image = tf.image.resize_image_with_crop_or_pad(image, 600, 600)
# box_image = tf.image.resize_images(image, [600, 600])
# box_image = tf.image.resize_images(box_image, [530, 399])
# with tf.Session() as sess:
#     box_image = sess.run(box_image)
# cv2.imwrite("box.jpeg", box_image)
# 
# 
# image = cv2.imread("1.JPEG")
# y = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
# square_image = tf.image.resize_images(y, [height, width])
# with tf.Session() as sess:
#     batch = sess.run(square_image)
#                     
