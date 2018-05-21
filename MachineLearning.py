
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate 100 points
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# W is dimension 1, between -1 and 1 
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b 

# loss
loss = tf.reduce_mean(tf.square(y - y_data))
# learningRate is 0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0 :
        print('step= ' + str(step), ' Weight= '+ str(sess.run(W)), ' baias= ' + str(sess.run(b)))
        plt.plot( x_data, y_data, 'ro', label='Original Data')
        plt.plot( x_data, sess.run(W) * x_data + sess.run(b), label='optimizer')
        plt.legend()
        plt.show()
        


# In[14]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[16]:


training = mnist.train.images
trainlabel = mnist.train.labels
nsample = 1 
randidx = np.random.randint(training.shape[0], size=nsample)

for i in [0,1,2]:
    curr_img = np.reshape(training[i, :], (28, 28))
    curr_label = np.argmax(trainlabel[i, :])
    plt.matshow( curr_img, cmap=plt.get_cmap('gray'))
    plt.title( "" + str(i + 1) + "th Training Data "+ "Label is " + str(curr_label))
    plt.show()


# In[19]:


# Softmax Regression
# tensorflow->placeholder( data_type, [ origin_dimension, transfer_dimension ])
x = tf.placeholder(tf.float32, [None, 784])
# 784 = 28 * 28

'''Setting Weight & biass'''
W = tf.Variable(tf.zeros([784, 10]))
# W x X dimension 784 to dimension 10 ex:0~9
b = tf.Variable(tf.zeros([10]))
# W x X + b dimension have to be 10
'''Setting Weight & biass'''


'''Regression of Softmax'''
y = tf.nn.softmax(tf.matmul(x, W)+ b)
'''Regression of Softmax'''


'''Cross-entropy'''
y_ = tf.placeholder(tf.float32, [None, 10])
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
'''Cross-entropy'''


'''training'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 0.5 is the learning rate
'''training'''


'''Run'''
init = tf.global_variables_initializer()
# session
sess = tf.Session()
# session initialize varriable
sess.run(init) 

for i in range(1000):
    test_x, test_y = mnist.train.next_batch(100)
    sess.run( train_step, feed_dict ={x :test_x, y_ :test_y})
'''Run'''


'''result'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print( sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''result'''
'''

# accuracy about 92%


# In[2]:


'''
'''Convolutional Neural Network'''
x = tf.placeholder(tf.float32, [None, 784])
# 784 = 28 * 28

'''Setting Weight & biass'''
W = tf.Variable(tf.zeros([784, 10]))
# W x X dimension 784 to dimension 10 ex:0~9
b = tf.Variable(tf.zeros([10]))
# W x X + b dimension have to be 10
'''Setting Weight & biass'''


'''Regression of Softmax'''
y = tf.nn.softmax(tf.matmul(x, W)+ b)
'''Regression of Softmax'''


'''Cross-entropy'''
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

'''layer 1 convolution'''
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
'''layer 1 convolution'''


'''layer 2 convolution'''
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
'''layer 2 convolution'''


'''Fully Connected'''
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
'''Fully Connected'''


'''Drop out'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout( h_fc1, keep_prob)
'''Drop out'''


'''layer out'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''layer out'''


'''run'''
sess = tf.Session()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # adam optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
'''run'''
'''


# In[ ]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
current_dir = os.getcwd()
sess = tf.InteractiveSession()

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable( initial, name)

def conv2d( X, W):
    return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def add_layer(X, W, B):
    h_conv = tf.nn.relu(conv2d(X,W)+ B)
    return max_pool_2x2(h_conv)

def pca(X, n_components):
    pca = PCA(n_components = n_components)
    pca.fit(X)
    return pca.transform(X)

def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40)
    return model.fit_transform(X)

def plot_scatter(x, labels, title, txt = False):
    plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    txts = []
    if txt:
        for i in range(10):
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.show()
    
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

layer1 = add_layer(x_image, weight_variable([5,5,1,32], "w_conv1"), bias_variable([32], "b_conv1"))
layer2 = tf.nn.relu( conv2d( layer1, weight_variable([5,5,32,48], "w_conv2"))+ bias_variable([48], "b_conv2"))
layer3 = add_layer(layer2, weight_variable([5,5,48,64], "w_conv3"), bias_variable([64], "b_conv3"))

W_fc1 = weight_variable([7*7*64, 1024], "w_fc1")
b_fc1 = bias_variable([1024], "b_fc1")
h_pool2_flat = tf.reshape(layer3, [-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10], "w_fc2")
b_fc2 = bias_variable([10], "b_fc2")
y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predition = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean( tf.cast( correct_predition, tf.float32))

saver=tf.train.Saver()
saver.restore(sess, os.path.join(current_dir, "model\mnist_cnn_3_layer\model.ckpt"))

test_size = 5000
test_data = mnist.test.images[0:test_size, :]
test_label = mnist.test.labels[0:test_size, :]
test_label_index = np.argmax(test_label, axis = 1)

layer1_reshape = tf.reshape(layer1[:, :, :, :], [-1, 14 * 14 * 32])
test_layer1_pca = pca(layer1_reshape.eval(feed_dict = {x: test_data}), 2)
plot_scatter(test_layer1_pca, test_label_index, "conv layer1 with pca")

