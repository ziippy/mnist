import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle

with open('ORENIST.data', 'rb') as file:
    images, labels = pickle.load(file, encoding="bytes")

# 원본 이미지
fig = plt.figure(figsize=(10,5))
for i in range(40):
    subplot = fig.add_subplot(4, 10, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(images[i].reshape(28,28), vmin=0, vmax=1,
                   cmap=plt.cm.gray_r, interpolation='nearest')

# 필터 정의
def edge_filter():
    filter0 = np.array(
        [[2, 1, 0, -1, -2],
         [3, 2, 0, -2, -3],
         [4, 3, 0, -3, -4],
         [3, 2, 0, -2, -3],
         [2, 1, 0, -1, -2]]) / 23.0
    filter1 = np.array(
        [[2, 3, 4, 3, 2],
         [1, 2, 3, 2, 1],
         [0, 0, 0, 0, 0],
         [-1, -2, -3, -2, -1],
         [-2, -3, -4, -3, -2]]) / 23.0

    filter_array = np.zeros([5, 5, 1, 2])
    filter_array[:, :, 0, 0] = filter0
    filter_array[:, :, 0, 1] = filter1

    return tf.constant(filter_array, dtype=tf.float32)

# 이미지 데이터에 필터 계산
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv = edge_filter()
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv,
                             strides=[1,1,1,1], padding='SAME'))
h_conv_cutoff = tf.nn.relu(h_conv-0.2)

h_pool =tf.nn.max_pool(h_conv_cutoff, ksize=[1,2,2,1],
                       strides=[1,2,2,1], padding='SAME')

plt.show()