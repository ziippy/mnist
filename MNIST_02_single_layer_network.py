import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 난수의 시드 생성
np.random.seed(20180121)

# MNIST 데이터 세트 다운로드
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

''' 단순히 matmul 만 이용한 경우 (MNIST_01_softmax_estimation.py 내용)
# 소프트맥스 함수에 의한 확률 p 계산식을 준비한다.
x = tf.placeholder(tf.float32, [None, 784])         # 28x28 사이즈이므로 784
w = tf.Variable(tf.zeros([784, 10]))                # 10개로 분류할 것이므로 (0으로 초기화)
b = tf.Variable(tf.zeros([10]))                    # bias (0으로 초기화)
f = tf.matmul(x, w) + b                            # 전 결합
p = tf.nn.softmax(f)                                # softmax
'''

num_units = 1024        # 은닉 계층의 노드 개수
x = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.truncated_normal([784, num_units]))
b1 = tf.Variable(tf.zeros([num_units]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)         # ReLU 이용

# 은닉 계층의 출력 결과로 소프트맥스 함수를 이용해 확률을 계산하는 부분
w0 = tf.Variable(tf.zeros([num_units, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

# 오차 함수와 트레이닝 알고리즘 준비
t = tf.placeholder(tf.float32, [None, 10])          # 트레이닝 세트의 라벨을 담기 위해 10
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

# 정답률 체크 정의
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션 준비
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 파라미터 최적화를 2000회 반복
i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)        # 미니배치
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
            feed_dict={x:mnist.test.images, t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

# 얻어진 결과를 실제 이미지로 확인
images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x:images, t: labels})

fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for (image, label, pred) in zip(images, labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c < 4 and i == actual) or (c >= 4 and i != actual):
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' % (prediction, actual))
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                           cmap=plt.cm.gray_r, interpolation="nearest")
            c += 1
            if c > 6:
                break

plt.show()

###
# Accuracy 결과는 약 0.97
# 출력 계층의 softmax 만 사용했을 때의 정답률 (0.92) 에 비해서는 크게 개선되었다고 할 수 있다.