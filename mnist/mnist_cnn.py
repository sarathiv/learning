import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
def create_weight(shape):
    noise = tf.truncated_normal(shape,
stddev=0.1)
    return tf.Variable(noise)

def create_bias(shape):
    small_constant = tf.constant(0.1,shape=shape)
    return tf.Variable(small_constant)

def conv2d(x_input,weight):
    strides = [1,1,1,1]
    return tf.nn.conv2d(x_input,weight,
strides=strides,padding='SAME')

def max_pool_2x2(x_input):
    ksize = [1,2,2,1]
    strides = [1,2,2,1]
    return tf.nn.max_pool(x_input,
ksize=ksize,strides=strides,padding='SAME')

if __name__ == '__main__':
    mnist_data = input_data.read_data_sets("MNIST_data/",
one_hot=True)
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    x_image = tf.reshape(x,[-1,28,28,1])

    #First set of conv,relu and max pooling
    weight1 = create_weight([5,5,1,32])
    bias1 = create_bias([32])
    conv_layer1 = conv2d(x_image,weight1) + bias1
    conv_layer1_relu = tf.nn.relu(conv_layer1)
    pool1 = max_pool_2x2(conv_layer1_relu)

    #second convmrelu and max pooling
    weight2 = create_weight([5,5,32,64])
    bias2 = create_bias([64])
    conv_layer2 = conv2d(pool1,weight2) + bias2
    conv_layer2_relu = tf.nn.relu(conv_layer2)
    pool2 = max_pool_2x2(conv_layer2_relu)

    #Fully connected layer
    weight_fcl = create_weight([7*7*64,1024])
    bias_fcl = create_bias([1024])
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    y_fcl = tf.matmul(pool2_flat,weight_fcl) + bias_fcl
    relu_y = tf.nn.relu(y_fcl)

    #Dropout layer
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(relu_y,keep_prob)

    #Read out layer
    weight_readout = create_weight([1024,10])
    bias_readout = create_bias([10])
    y_readout = tf.matmul(dropout,
weight_readout) + bias_readout

    cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(
labels=y_,logits=y_readout))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_readout,1),
tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loop_start_time = time.time()
        for i in range(20000):
            iteration_start_time = time.time()
            batch = mnist_data.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(
            feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
                print ('step %d , training accuracy %g' %(i,train_accuracy))

            train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
            print ('test accuracy %g' %
        accuracy.eval(feed_dict={
                    x: mnist_data.test.images,
                    y_: mnist_data.test.labels,
                    keep_prob: 1.0
                }))
            iteration_time_seconds = time.time()-iteration_start_time
            print ('Iterarion %d, %f minutes since start, loop iteration time %f seconds' %(i,(float)(time.time() - loop_start_time)/60,iteration_time_seconds))
            time_left_seconds = (float)(20000-i)/(iteration_time_seconds)
            print ('Time left in minutes : %f at %f per iterations ' %(time_left_seconds/60,time_left_seconds))
        print("CNN finished")
