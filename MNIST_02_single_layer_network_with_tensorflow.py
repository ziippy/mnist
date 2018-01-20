import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 난수의 시드 생성
np.random.seed(20180121)

# MNIST 데이터 세트 다운로드
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 단층 신경망을 표현하는 클래스 정의
class SingleLayerNetwork:
    def __init__(self, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_units)
            self.prepare_session()

    def prepare_model(self, num_units):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')

        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([784, num_units]), name='weights')
            b1 = tf.Variable(tf.zeros([num_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # 단일 값에 대해서는 summary.scalar
        # 복수의 요소를 포함하는 것에 대해서는 summary.histogram
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", w1)
        tf.summary.histogram("biases_hidden", b1)
        tf.summary.histogram("weights_output", w0)
        tf.summary.histogram("biases_output", b0)

        # 클래스 외부에서 참조할 필요가 있는 변수를 인스턴스 변수로 공개
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist/02_sln_logs", sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer

# 신경망 인스턴스 생성 후 이를 사용
nn = SingleLayerNetwork(1024)

# 파라미터 최적화를 2000회 반복
i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)        # 미니배치
    #sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    if i % 100 == 0:
        #loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, t: mnist.test.labels})
        summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy], feed_dict={nn.x: mnist.test.images, nn.t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

        # For, tensorboard
        nn.writer.add_summary(summary, i)

# 얻어진 결과를 실제 이미지로 확인
images, labels = mnist.test.images, mnist.test.labels
#p_val = sess.run(p, feed_dict={x:images, t: labels})
p_val = nn.sess.run(nn.p, feed_dict={nn.x:images, nn.t: labels})

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