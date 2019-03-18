import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

fd = open('train-images-idx3-ubyte')
train_images = np.fromfile(file=fd, dtype=np.uint8)
train_images = train_images[16:].reshape([60000, 28, 28]).astype(np.float)
train_images = train_images / 255.0

fd = open('train-labels-idx1-ubyte')
train_labels = np.fromfile(file=fd, dtype=np.uint8)
train_labels = train_labels[8:].reshape([60000]).astype(np.int)

X = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32, name='X')
Y = tf.placeholder(shape=[None], dtype=tf.int64, name='Y')
W = tf.Variable(tf.zeros([28*28, 10]))
B = tf.Variable(tf.zeros([10]))
Y_onehot = tf.one_hot(Y, 10, axis=1)
Y_ = tf.matmul(X, W) + B

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y_onehot))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
optimize = trainer.minimize(loss)

softmax = tf.nn.softmax(Y_)
correct = tf.equal(tf.argmax(softmax, 1), tf.argmax(Y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

fd = open('t10k-images-idx3-ubyte')
test_images = np.fromfile(file=fd, dtype=np.uint8)
test_images = test_images[16:].reshape([10000, 28*28]).astype(np.float)
test_images = test_images / 255.0

fd = open('t10k-labels-idx1-ubyte')
test_labels = np.fromfile(file=fd, dtype=np.uint8)
test_labels = test_labels[8:].reshape([10000]).astype(np.int)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 10000
    batch_count = 60000 // batch_size

    for epoch in range(50):
        total_loss = 0
        train_total_acc = 0
        for i in range(batch_count):
            batch_index = i * batch_size

            img = np.reshape(train_images[batch_index: batch_index + batch_size], [batch_size, 28 * 28])

            label = train_labels[batch_index: batch_index + batch_size]
            _, loss_v = sess.run([optimize, loss], feed_dict={X: img, Y: label})
            total_loss += loss_v

            train_acc = sess.run(accuracy, feed_dict={X: img, Y: label})
            train_total_acc += train_acc

        if epoch % 10 == 0:
            print (train_total_acc / batch_count)

    w = sess.run(W)

    for i in range(10):
        w_ = w[:, i]
        print (w_.shape)
        min = w_.min()
        max = w_.max()
        w_ = (w_ - min) * (255.0 / (max - min))

        # plt.imshow(np.reshape(w_, [28, 28]))
        plt.imsave('./images/image%s.png' % i, np.reshape(w_, [28, 28]), cmap=plt.cm.gray)

    test_acc = sess.run(accuracy, feed_dict={X: test_images, Y: test_labels})
    # print 'test accuracy : ', test_acc