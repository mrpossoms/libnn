# This file is part of libnn.
#
# libnn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libnn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libnn.  If not, see <https://www.gnu.org/licenses/>.

import struct
import numpy as np
import tensorflow as tf
from helpers import serialize_matrix

# tf.set_random_seed(0) # make libnn debugging easier

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def to_one_hot(labels):
    Y = np.zeros((labels.shape[0], 10))

    for yi in range(0, labels.shape[0]):
        Y[yi][labels[yi]] = 1
    return Y


def ts(root):
    X = (read_idx(root + '/images-idx3-ubyte') / 255.0)
    Y = to_one_hot(read_idx(root + '/labels-idx1-ubyte'))

    return X.reshape((X.shape[0], 28 * 28)), Y

ts_X, ts_Y = ts('/Users/kirk/code/nn.h/data/model_conv2/ds/train')

from PIL import Image

i = np.random.randint(0, ts_X.shape[0])
print(i)
img = (ts_X[i].reshape((28, 28)) * 255).astype(np.uint8)
img = Image.fromarray(img, 'L')
img.show(title=str(ts_Y[i]))


X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

P = {
    'c0_kernel': tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1)),
    'c0_bias':   tf.Variable(tf.constant(0.1, shape=[32])),

    'c1_kernel': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    'c1_bias': tf.Variable(tf.constant(0.1, shape=[64])),

    'c2_kernel': tf.Variable(tf.truncated_normal([5, 5, 64, 10])),
    'c2_bias': tf.Variable(tf.constant(0.1, shape=[1, 10])),
}

x_image = tf.reshape(X, [-1, 28, 28, 1])

# Layer 0
c0_z = tf.nn.conv2d(x_image, P['c0_kernel'], [1, 1, 1, 1], 'VALID') + P['c0_bias']
c0_a = tf.nn.relu(c0_z)
c0_p = tf.nn.max_pool(c0_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
a = c0_p

# Layer 1
c1_z = tf.nn.conv2d(a, P['c1_kernel'], [1, 1, 1, 1], 'VALID') + P['c1_bias']
c1_a = tf.nn.relu(c1_z)
c1_p = tf.nn.max_pool(c1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
a = c1_p

# Layer 2
c2_z = tf.nn.conv2d(a, P['c2_kernel'], [1, 1, 1, 1], 'VALID') + P['c2_bias']
h = tf.reshape(c2_z, [-1, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# run gradient descent to fit parameters
batch_size = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(3000):
    i = np.random.randint(0, ts_X.shape[0] - batch_size)
    sub_ts_X = ts_X[i:i + batch_size]
    sub_ts_Y = ts_Y[i:i + batch_size]

    sess.run(optimize, feed_dict={X: sub_ts_X, Y: sub_ts_Y})
    if e % 100 == 0:
        train_h = sess.run(h, feed_dict={X: sub_ts_X, Y: sub_ts_Y})
        # train_prediction = sess.run(correct_prediction, feed_dict={X: sub_ts_X, Y: sub_ts_Y})
        train_accuracy = sess.run(accuracy, feed_dict={X: sub_ts_X, Y: sub_ts_Y})
        print('step %d, training accuracy %f' % (e, train_accuracy))

# Save the learned parameters
for key in P:
    file_name = key.replace('_', '.')

    with open('/var/model/' + file_name, mode='wb') as fp:
        serialize_matrix(sess.run(P[key]), fp)

# test
print('Done, evaluating')
ts_X, ts_Y = ts('/Users/kirk/code/nn.h/data/model_conv2/ds/test')

correct = 0
incorrect = 0
for i in range(0, ts_X.shape[0]):
    hi = sess.run(h, feed_dict={X: ts_X[i].reshape((1, 784)) })
    if hi.argmax() == ts_Y[i].argmax():
        correct += 1
    else:
        incorrect += 1

print("%d / %d (right/wrong)" % (correct, incorrect))
print("%f%% accuracy" % (correct / ts_X.shape[0]))