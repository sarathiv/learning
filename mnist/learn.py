import tensorflow as tf

a = tf.Variable([2000000],dtype=tf.float64)
x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)
calculation = x + a
loss = tf.reduce_sum(tf.square(y-calculation))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(500):
    sess.run(train,{x:[50,52,54,56],y:[150,152,154,156]})
    print(str(i) + " " + str(sess.run([a])))
