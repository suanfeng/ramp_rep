#coding=utf-8

import input_data_nd
import tensorflow as tf

mnist = input_data_nd.read_data_sets("../../data/MNIST/", one_hot=True)

"""x 代表任意数量的MNIST图标,None代表图像的数量"""
x = tf.placeholder(tf.float32, [None, 784])

"""W,代表权重,10个数字,每个数字在每个像素上的权重"""
W = tf.Variable(tf.zeros([784, 10]))

"""10个数字,每个数字的偏置量"""
b = tf.Variable(tf.zeros([10]))

"""softmax 回归,y为预测概率分布"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""y_代表实际的概率分布"""
y_ = tf.placeholder("float", [None, 10])

"""交叉熵算法"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

"""用梯度下降最小交叉熵"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

"""每次用100个随机训练训练1000次"""
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuray = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuray, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
