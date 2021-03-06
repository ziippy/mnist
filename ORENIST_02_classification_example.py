import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle

# 난수의 시드 생성
np.random.seed(20180127)
tf.set_random_seed(20180127)

# 데이터 파일 읽기
with open('ORENIST.data', 'rb') as file:
    images, labels = pickle.load(file, encoding="bytes")

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

# 신경망 정의
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

#W_conv = edge_filter()                                             # static filter
W_conv = tf.Variable(tf.truncated_normal([5,5,1,2], stddev=0.1))    # dynamice filter
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv,                       # 합성곱 필터를 적용하는 함수인 conv2d, 절대값을 취하는 함수인 abs
                             strides=[1,1,1,1], padding='SAME'))    # SAME 은 존재하지 않는 부분의 픽셀은 0 으로 해서 계산
h_conv_cutoff = tf.nn.relu(h_conv-0.2)      # 필터의 효과를 강조해서 알기 쉽게 하기 위해 추가

h_pool =tf.nn.max_pool(h_conv_cutoff, ksize=[1,2,2,1],
                       strides=[1,2,2,1], padding='SAME')

# 풀링 계층의 출력을 두 개의 노드로 된 전 결합층에 입력하고, 소프트맥스를 통해 3가지 데이터로 분류
h_pool_flat = tf.reshape(h_pool, [-1, 392])     # 14*14*2

num_units1 = 392
num_units2 = 2

# 전 결합층
w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(h_pool_flat, w2) + b2)

# 출력 층
w0 = tf.Variable(tf.zeros([num_units2, 3]))
b0 = tf.Variable(tf.zeros([3]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

######## 오차 함수, 트레이닝 알고리즘, 정답률 정의
t = tf.placeholder(tf.float32, [None, 3])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##### 세션을 정의하고
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

##### 파라미터 최적화를 200회 반복
i = 0
for _ in range(200):
    i += 1
    sess.run(train_step, feed_dict={x:images, t:labels})
    if i % 10 == 0:
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={x:images, t:labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

# 각각의 데이터의 특징 변수 (z1, z2)를 산포로도 표시
hidden2_vals = sess.run(hidden2, feed_dict={x: images})

z1_vals = [[], [], []]
z2_vals = [[], [], []]

for hidden2_val, label in zip(hidden2_vals, labels):
    label_num = np.argmax(label)
    z1_vals[label_num].append(hidden2_val[0])
    z2_vals[label_num].append(hidden2_val[1])

fig = plt.figure(figsize=(5, 5))
subplot = fig.add_subplot(1, 1, 1)
subplot.scatter(z1_vals[0], z2_vals[0], s=200, marker='|')
subplot.scatter(z1_vals[1], z2_vals[1], s=200, marker='_')
subplot.scatter(z1_vals[2], z2_vals[2], s=200, marker='+')

plt.show()
