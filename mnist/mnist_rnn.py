import tensorflow as tf
import time
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
num_features = 128
timesteps = 28
timestep_input = 28
def create_dicts(words):
    index = 0
    dictionary = {}
    reverse_dictionary = {}
    for word in words:
        dictionary[word] = index
        reverse_dictionary[index] = word
        index += 1
    return dictionary,reverse_dictionary

def create_lstm_layer(x,weight,bias,keep_prob):
    lstm_x = tf.unstack(x,timesteps,1)
    lstm_cell = rnn.BasicLSTMCell(num_features,forget_bias = 1.0)
    dropout_lstm = rnn.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
    outputs, _ = rnn.static_rnn(dropout_lstm,lstm_x,dtype=tf.float32)
    output_x = outputs[-1]
    return tf.matmul(output_x,weight) + bias

if __name__ == "__main__":
    mnist_data = input_data.read_data_sets("MNIST_data",one_hot=True)
    x = tf.placeholder(tf.float32,[None,timesteps,timestep_input])
    W = tf.Variable(tf.random_normal([num_features,10]))
    b = tf.Variable(tf.random_normal([10]))
    keep_prob = tf.placeholder(tf.float32)
    y = create_lstm_layer(x,W,b,keep_prob)
    y_ = tf.placeholder(tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        loop_start_time = time.time()
        iteration_start_time = time.time()
        for i in range(30000):
            batch_xs, batch_ys = mnist_data.train.next_batch(128)
            batch_xs = batch_xs.reshape([128,timesteps,timestep_input])
            sess.run(train,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
            if i%200 == 0:
                iteration_time_seconds = time.time()-iteration_start_time
                print ('Iterarion %d, %f minutes since start, 200 iteration time %f seconds' %(i,(float)(time.time() - loop_start_time)/60,iteration_time_seconds))
                time_left_seconds = (float)(30000-i)*(iteration_time_seconds/200)
                print ('Time left in minutes : %f at %f per iterations ' %(time_left_seconds/60,time_left_seconds))
                iteration_start_time = time.time()
                train_accuracy = accuracy.eval(
            feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
                print('step %d , training accuracy %g' %(i,train_accuracy))
                reshaped_test_image = mnist_data.test.images[:].reshape(([-1,timesteps,timestep_input]))
                print('LSTM accuracy %g ' %
            accuracy.eval(feed_dict={
                x:reshaped_test_image,
                y_:mnist_data.test.labels,
                keep_prob:1.0}))
        saver.save(sess,"model/rnn/mnist_rnn")
        print('RNN Finished...')
