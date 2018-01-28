import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 난수의 시드 생성
np.random.seed(20180127)

# MNIST 데이터 세트 다운로드
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 다층 신경망을 표현하는 클래스 정의
class DoubleCnnNetwork:
    def __init__(self, num_filters1, num_filters2):
        with tf.Graph().as_default():
            self.prepare_model(num_filters1, num_filters2)
            self.prepare_session()

    def prepare_model(self, num_filters1, num_filters2):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')
            x_image = tf.reshape(x, [-1,28,28,1])

        with tf.name_scope('conv'):
            w_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters1], stddev=0.1))
            h_conv = tf.nn.conv2d(x_image, w_conv, strides=[1,1,1,1], padding='SAME')
            b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
            h_conv_cutoff = tf.nn.relu(h_conv + b_conv)
            h_pool = tf.nn.max_pool(h_conv_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        with tf.name_scope('conv2'):
            w_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filters1,num_filters2], stddev=0.1))
            h_conv2 = tf.nn.conv2d(h_pool, w_conv2, strides=[1,1,1,1], padding='SAME')
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
            h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        with tf.name_scope('fully-connection'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])

            num_units1 = 7*7*num_filters2
            num_units2 = 1024

            w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
            b2 = tf.Variable(tf.zeros([num_units2]))
            hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units2, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # 단일 값에 대해서는 summary.scalar
        # 복수의 요소를 포함하는 것에 대해서는 summary.histogram
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        #tf.summary.histogram("weights_hidden", w1)
        #tf.summary.histogram("biases_hidden", b1)
        #tf.summary.histogram("weights_output", w0)
        #tf.summary.histogram("biases_output", b0)

        # 클래스 외부에서 참조할 필요가 있는 변수를 인스턴스 변수로 공개
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.restore(sess, './session/cnn_session-4000')
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist/03_cnn_logs", sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver

# 단층 CNN 인스턴스 생성 후 이를 사용
nn = DoubleCnnNetwork(32, 64)     # 32, 64 is 필터 수

# 파라미터 최적화를 20000회 반복
i = 0
for _ in range(20000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(50)        # 미니배치
    #sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    if i % 500 == 0:
        #loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, t: mnist.test.labels})
        summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy], feed_dict={nn.x: mnist.test.images, nn.t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

        nn.saver.save(nn.sess, './session/double_cnn_session', global_step=i)

        # For, tensorboard
        nn.writer.add_summary(summary, i)

'''
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
'''

###
# Accuracy 결과는 약 0.981
# 출력 계층의 softmax 만 사용했을 때의 정답률 (0.92) 에 비해서는 크게 개선되었다고 할 수 있다.
# 신경망을 이용했을 때의 정답률 (0.97) 에 비해서 조금 개선되었다고 할 수 있다.