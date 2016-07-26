import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt('train2.txt',unpack=True,dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))
W = tf.placeholder(tf.float32)

h = tf.matmul(W,X)
hypothesis = tf.div(1.,1.+tf.exp(-h))
 
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.1)
#optimizer = tf.train.GradientDescentOptimizer(a)
#train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
costes = []
Ws = []
for i in range(-500,501):
	a.assign( i* 0.01)
	costes.append(sess.run(cost,{W:a,X:x_data,Y:y_data}))
	Ws.append(sess.run(a))
#for step in xrange(10000):
#	sess.run(train,feed_dict={X:x_data,Y:y_data})
#	if step % 20 == 0:
#		print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W)

plt.plot(Ws,costes,'ro')
plt.xlabel('W')
plt.ylabel('Cost')
plt.savefig('/root/Server/public/ftp/plot.png')
