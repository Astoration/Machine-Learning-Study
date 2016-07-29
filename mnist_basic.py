import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32, [None,784]) # Bitmax
y_ = tf.placeholder(tf.float32, [None, 10]) #Answer 0~9

W = tf.Variable(tf.zeros([784,10])) # Weight for fully connected layer #1
b = tf.Variable(tf.zeros([10])) #Bias for fully connected layer #1

# Fully connected layer -> Softmax
y = tf.nn.softmax(tf.matmul(x,W)+b) #Our guess
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])) # Cost function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #Train with GradientDescentOptimizer

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # How many has matched
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1001):
		batch_xs, batch_ys = mnist.train.next_batch(500) # Batch 100 images
		sess.run(train_step,feed_dict={x: batch_xs, y_ : batch_ys}) # Run training with batch
		if i % 100 == 0: # For each 100 steps, print out current accuracy
			print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_ : mnist.test.labels}))
	print sess.run(tf.argmax(sess.run(y, feed_dict={x:mnist.test.images[0:100]}),1))
