import tensorflow as tf
import numpy as np


def example(label):
    x = np.zeros((9, 9, 1))
    y_int = np.random.randint(-3, 3)

    if label is 0:
        return np.random.random_sample((9, 9, 1)) - 0.5

    for yi in range(0, 9):
        for xi in range(0, 9):
            _y = xi + y_int

            if label is 1:
                x[yi][xi][0] = 1.0 if yi < _y else 0
            if label is 2:
                x[yi][xi][0] = 1.0 if yi > _y else 0

    return x


def one_hot(label):
    y = np.zeros((1, 1, 3))
    y[0][0][label] = 1
    # y = np.zeros((3))
    # y[label] = 1
    return y


def ts(n=100):
    X = []
    Y = []

    for _ in range(n):
        label = np.random.randint(0, 3)
        X += [example(label)]
        Y += [one_hot(label)]

    return np.array(X), np.array(Y)

ts_X, ts_Y = ts()


X = tf.placeholder(tf.float32, shape=[None, 9, 9, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1, 1, 3])

P = {
    'c0_kernel': tf.Variable(tf.truncated_normal([3, 3, 1, 3])) * 0.01,
    'c0_bias'  : tf.Variable(tf.constant(0.1, shape=[3])),
    'c1_kernel': tf.Variable(tf.truncated_normal([7, 7, 3, 3])) * 0.01,
    'c1_bias'  : tf.Variable(tf.constant(0.1, shape=[3]))
}

c0_z = tf.nn.conv2d(X, P['c0_kernel'], [1, 1, 1, 1], 'VALID') + P['c0_bias']
c0_a = tf.nn.relu(c0_z)

c1_z = tf.nn.conv2d(c0_a, P['c1_kernel'], [1, 1, 1, 1], 'VALID') + P['c1_bias']
p = tf.nn.softmax(c1_z)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=p)
loss = tf.reduce_mean(cross_entropy) + tf.nn.l2_loss(P['c0_kernel']) + tf.nn.l2_loss(P['c0_kernel'])

optimize = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(1000):
    sess.run(optimize, feed_dict={X: ts_X, Y: ts_Y})

    if e % 100 == 0:
        print(sess.run(loss, feed_dict={X: ts_X, Y: ts_Y}))

# If learning was successful, should print a rank-3 identity matrix
for c in range(3):
    p_ = sess.run(p, feed_dict={X: np.array([example(c)])})
    print((p_ >= p_.max()) * 1)


def serialize_matrix(m, fp):
    import struct

    # write the header
    fp.write(struct.pack('b', len(m.shape)))
    for d in m.shape:
        fp.write(struct.pack('i', d))

    # followed by each element
    for e in m.flatten():
        fp.write(struct.pack('f', e))

# Save the learned parameters
for key in P:
    file_name = key.replace('_', '.')

    with open('/var/model/' + file_name, mode='wb') as fp:
        if 'kernel' in file_name:
            k = sess.run(P[key])
            w, h, i, o = k.shape
            serialize_matrix(k, fp)
        elif 'bias' in file_name:
            serialize_matrix(sess.run(P[key]), fp)

c0_w = sess.run(P['c0_kernel'])

class0 = np.array([[0.4373552677664668,0.4779668005977227,0.1930176971271862,0.15935195050556805,-0.03454942695534857,0.022374068795748436,0.06405680026965888,0.39244537551760794,-0.2735623389910957,0.38858326609909133,0.2389487947151785,0.48374378419428665,-0.2873746681814301,0.15689272126594211,-0.2247380377239223,-0.36435552100792035,-0.113894519160612,-0.433816069482448,-0.047338430355342576,-0.0074328193130708264,0.25158455017447046,-0.1316869118355698,0.32732382729263276,-0.03149328270894902,0.40673487883180215,0.23059401025303827,-0.14761593512965798,-0.489243676698937,0.4205509690622199,0.307691127548238,0.19089984721281394,0.39117070269955867,-0.32715124571459187,0.1411097709774165,0.10011676261464031,0.005543642303233787,0.12302408086427319,-0.42839800593512833,-0.03371917793608259,0.44079954354622175,-0.19602342865999123,0.44025648310799415,0.4663607283882778,0.10492501453134695,-0.34127869123826937,-0.18989831312044736,0.2029763412847888,0.2215775047424121,0.22286667629166013,0.17640028827130871,0.20423948382178803,-0.11416520245672268,0.4351949295559211,0.16930399873131985,-0.3953896478574027,-0.05084905409556473,-0.25921158228950436,-0.14211563779966496,-0.42265289391845084,-0.36933147427508617,-0.49382288624234916,-0.044746257776212994,-0.014855572304796283,0.4080228481416096,-0.3220690518534908,0.1394160420002314,0.09462338821082628,0.2779240661556387,0.26627358437614124,0.1280167941356074,-0.4852490528273933,0.3672766997201402,0.24093681779077813,0.4829553576078329,-0.40584452872711907,-0.2034742481867905,-0.13570583777988487,0.1493635626701023,-0.17886701602813115,0.19386606894576586,-0.36245899007683546]
]).reshape(1,9,9,1)

class2 = np.array([
    0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
]).reshape((1,9,9,1)) + 0.5

h0 = sess.run(p, feed_dict={X: class0})
print("Class0 prediction")
print(h0)


h2 = sess.run(p, feed_dict={X: class2})
print("Class2 prediction")
print(h2)

# m_c0_z = sess.run(c0_z, feed_dict={X: class3})
# m_c0_w = sess.run(P['c0_kernel'])
# m_c0_b = sess.run(P['c0_bias'])
#
# print(m_c0_w)
# print(m_c0_b)
#
# print(m_c0_z.shape)
# print(m_c0_z[0][0][0])
# print(m_c0_z[0])


# for c in range(3):
#     s = ''
#     for e in example(c).flatten():
#         s += str(e) + ','
#     print(s)