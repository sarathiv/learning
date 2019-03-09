import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

sess = tf. Session()
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs",sess.graph)
sess.run(init)

for i in range(1000):
        batch_xs,batch_ys = mnist_data.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(
    accuracy,
    feed_dict={x:mnist_data.test.images,
    y_:mnist_data.test.labels}))
