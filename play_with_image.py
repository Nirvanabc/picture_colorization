import tensorflow as tf
import numpy as np
import cv2
from constants import *


## FIXME!
# 1) It should give any size any time, not set in
# the beginning
# 2) It should return array of real sizes to print real image
# if I want

def get_batch(batch_size):
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
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
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

