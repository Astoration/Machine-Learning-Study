import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32, [None,784]) # Bitmax
y_ = tf.placeholder(tf.float32, [None, 10]) #Answer 0~9

W = tf.Variable(tf.random_normal([784,256])) # Weight for fully connected layer #1
b = tf.Variable(tf.random_normal([256])) #Bias for fully connected layer #1

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))

# Fully connected layer -> Softmax
L1 =tf.nn.relu(tf.matmul(x,W)+b) #Our guess
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
y = tf.matmul(L2,W3)+b3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_)) # Cost function
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy) #Train with GradientDescentOptimizer

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # How many has matched
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in range(10001):
		batch_xs, batch_ys = mnist.train.next_batch(100) # Batch 100 images
		sess.run(train_step,feed_dict={x: batch_xs, y_ : batch_ys}) # Run training with batch
		if i % 100 == 0: # For each 100 steps, print out current accuracy
			print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_ : mnist.test.labels}))
	print sess.run(tf.argmax(sess.run(y, feed_dict={x:mnist.test.images[0:100]}),1))
