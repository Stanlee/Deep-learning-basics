import tensorflow as tf
from fire_theft import *

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
B = tf.Variable(0, dtype=tf.float32)
W = tf.Variable(0, dtype=tf.float32)
Y_pred = W*X + B

loss = tf.square(Y_pred - Y)
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        total_loss = 0.0
        for x, y in fire_theft_data:
            _, err = sess.run([updateModel, loss], feed_dict={X: x, Y: y})
            total_loss += err

        print ('trained value: w=%s, b=%s loss=%s' % (sess.run(W), sess.run(B), total_loss))
        # print 'loss : %s' % total_loss