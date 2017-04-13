
# coding: utf-8

# In[43]:
from __future__ import print_function, division
from read_images import read_labeled_image_list, read_images_from_disk
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
import numpy as np
from sklearn.manifold import TSNE
##get_ipython().magic('matplotlib inline')
#get_ipython().magic('pylab inline')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
random.seed(1337)

# from tensorflow import ops
# from tensorflow import dtypes

# In[44]:
# device_name = sys.argv[1]
# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"

#with tf.device('/gpu:0'):
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# I    n[45]:
# print "Asdfasdf"
learning_rate = 0.005
training_epochs = 150
batch_size = tf.placeholder(tf.int32,shape= [])
b_size = batch_size
display_step = 1
# logs_path = './tensorflow_logs/mnist_metrics'
# Network Parameters
the_size = 256
n_hidden_1 = 64#128 # 1st layer number of features
n_hidden_2 = 64#128# 2nd layer number of features
n_input = [the_size,the_size,1] #Alexnet relu_5 output dimensions
n_classes = 46 # DTD total classes (0-9 digits)
margin = 1000

    
    
    # In[46]:
    
x_left = tf.placeholder(tf.float32, shape=[None,n_input[0],n_input[1],1], name='InputDataLeft')
x_right = tf.placeholder(tf.float32, shape=[None,n_input[0],n_input[1],1], name='InputDataRight')
final_label = tf.placeholder(tf.float32, shape=[None, 1], name='LabelData') # 0 if the same, 1 is different
    
x_image_left = x_left
x_image_right = x_right
    
    
    # In[47]:
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def tfNN(x):
    x = tf.scalar_mul(1.0/256.0, x)
    x = tf.reshape(x,[-1,the_size*the_size])
    #x = tf.nn.l2_normalize(x,dim=1)
    #x = tf.sign(x)*tf.sqrt(tf.abs(x))
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    out_layer = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
    return out_layer#y_conv
    
n_input=the_size*the_size
## In[49]:
weights = {
'w1': tf.Variable(tf.random_uniform([the_size*the_size, n_hidden_1], minval=-4*np.sqrt(6.0/(n_input + n_hidden_1)), maxval=4*np.sqrt(6.0/(n_input + n_hidden_1))), name='W1'),
'w2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], minval=-4*np.sqrt(6.0/(n_hidden_1 + n_hidden_2)), maxval=4*np.sqrt(6.0/(n_hidden_1 + n_hidden_2))), name='W2'),
'w3': tf.Variable(tf.random_uniform([n_hidden_2, n_classes], minval=-4*np.sqrt(6.0/(n_hidden_2 + n_classes)), maxval=4*np.sqrt(6.0/(n_hidden_2 + n_classes))), name='W3'),
'w4': tf.Variable(tf.random_uniform([n_classes, 25], minval=-4*np.sqrt(6.0/(n_classes +
25)),maxval=4*np.sqrt(6.0/(n_classes + 25))), name='W4')
}
biases = {
'b1': tf.Variable(tf.truncated_normal([n_hidden_1]) / sqrt(n_hidden_1), name='b1'),
'b2': tf.Variable(tf.truncated_normal([n_hidden_2]) / sqrt(n_hidden_2), name='b2'),
'b3': tf.Variable(tf.truncated_normal([n_classes]) / sqrt(n_classes), name='b3'),
'b4': tf.Variable(tf.truncated_normal([25]) / sqrt(2), name='b4')#TODO: Fix normalization
}


# In[50]:

with tf.name_scope('Model'):
    # Model
    pred_left = tfNN(x_image_left)
    pred_right = tfNN(x_image_right)
    print(pred_right.get_shape())
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        #TODO: Keep dims??? Is reduce_indices correct??
        #pred_left = tf.Print(pred_left,[pred_left],'pred_left = ')
        #pred_right = tf.Print(pred_right,[pred_right],'pred_right = ')
        d = tf.reduce_sum(tf.square(pred_left - pred_right), 1, keep_dims=True)
        #d = tf.Print(d,[d],'unrooted = ')
        d_sqrt = tf.sqrt(d)
        #d_sqrt = tf.Print(d_sqrt,[margin-d_sqrt])
        #d_sqrt = tf.Print(d_sqrt, [d_sqrt], 'rooted = ')
        loss = final_label * tf.square(tf.maximum(0.0, margin - d_sqrt)) + (1 - final_label) * d
        loss = 0.5 * tf.reduce_sum(loss)#mean-->sum
 
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# In[51]:

# Initializing the variables

# Create a summary to monitor cost tensor
#tf.scalar_summary("loss", loss)

# Merge all summaries into a single op
# merged_summary_op = tf.merge_all_summaries()







filename = '../train_DTD_PATHS.txt'
label_file = '../DTD_LABELS.txt'
image_list, label_list = read_labeled_image_list(filename, label_file)

images = tf.convert_to_tensor(image_list, dtype=tf.string) #, dtype=tf.string)
labels = tf.convert_to_tensor(label_list) #, dtype=tf.int32)

input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs = 10*training_epochs,
                                            shuffle = False)
image, label = read_images_from_disk(input_queue=input_queue)


#should this be shuffle_batch?
images_for_train, labels_for_train = tf.train.batch([image, label], batch_size =batch_size,num_threads=1)#,capacity=10000,min_after_dequeue = 6000,allow_smaller_final_batch=True)

val_filename = '../val_DTD_PATHS.txt'

val_image_list, val_label_list = read_labeled_image_list(val_filename, label_file)

val_images = tf.convert_to_tensor(val_image_list, dtype=tf.string) #, dtype=tf.string)
val_labels = tf.convert_to_tensor(val_label_list) #, dtype=tf.int32)

val_input_queue = tf.train.slice_input_producer([val_images, val_labels],
                                            num_epochs = 10*training_epochs,
                                            shuffle = True)
val_image, val_label = read_images_from_disk(input_queue=val_input_queue)




images_for_val, labels_for_val = tf.train.batch([val_image, val_label],batch_size=700,num_threads=4)#,capacity=10000,min_after_dequeue = 200,allow_smaller_final_batch=True)


loss_tracker = []
# Launch the graph
init = tf.initialize_all_variables()
sess = tf.Session()#config = tf.ConfigProto(log_device_placement=True))#,allow_soft_placement = True))
sess.run(init)
sess.run(tf.initialize_local_variables())
tf.train.start_queue_runners(sess = sess)
# op to write logs to Tensorboard
#summary_writer = tf.train.FileWriter(logs_path, graph=tf.get_default_graph())
N=50

# Training cycle
for epoch in range(training_epochs):
    avg_loss = 0.0
    print (epoch)
    #if epoch%2==0:
    #    N=65
    #else:
    #    N=35
    batch_s = sess.run(b_size,feed_dict = {batch_size: N})
    total_batch = int(4000/ batch_s)
    # Loop over all batches
    for i in range(total_batch):
        samecount,diffcount = 0,0
        # print(i)
        # left_batch_xs, left_batch_ys = mnist.train.next_batch(batch_size)
        # right_batch_xs, right_batch_ys = mnist.train.next_batch(batch_size)
        left_batch_xs, left_batch_ys = sess.run([images_for_train,labels_for_train],feed_dict ={batch_size: N})
        right_batch_xs, right_batch_ys = sess.run([images_for_train,labels_for_train],feed_dict ={batch_size: N})
        y_labels = np.zeros((batch_s, 1))
        for l in range(batch_s):
            # print(l)
            if left_batch_ys[l] == right_batch_ys[l]:
                y_labels[l, 0] = 0.0
                samecount+=1
            else:
                y_labels[l, 0] = 1.0
                diffcount+=1
            #print(y_labels[l,0])
        print(samecount,diffcount)
        _, l = sess.run([optimizer, loss],
                                 feed_dict = {
                                              x_left: left_batch_xs,
                                              x_right: right_batch_xs,
                                              final_label: y_labels,
                                              batch_size: N
                                             })
        # Write logs at every iteration
#         summary_writer.add_summary(summary, epoch * total_batch + i)
        # Compute average loss

        avg_loss += l / total_batch
        #if i%100==0:
        #    print(avg_loss)
    # Display logs per epoch step
    if avg_loss <= 100:
        break
    if (epoch+1) % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "loss =", "{:.9f}".format(avg_loss))

    print ("Optimization Finished!")

    print ("Run the command line:\n"       "--> tensorboard --logdir=./tensorflow_logs "       "\nThen open http://0.0.0.0:6006/ into your web browser")


# In[ ]:

# Test model
# Calculate accuracy
for i in range(5):
    test_xs, test_ys = sess.run([images_for_train,labels_for_train],feed_dict={batch_size:700}) 
    ans = sess.run([pred_left], feed_dict = { x_left: test_xs})
    print(ans[0].shape)
    # In[ ]:


    np.save("ans" + str(i) + ".npy",ans)
    np.save("labs" + str(i)+ ".npy",test_ys)
    ans = ans[0]

test_xs, test_ys = sess.run([images_for_val,labels_for_val]) 
ans = sess.run([pred_left], feed_dict = { x_left: test_xs})
print(ans[0].shape)
# In[ ]:


np.save("ansval.npy",ans)
np.save("labsval.npy",test_ys)
ans = ans[0]
# In[ ]:

plt.figure(figsize=(3,3))
# scatter(r[:,0], r[:,1], c=[test_ys[x,:].argmax() for x in range(len(test_ys))])
plt.scatter(ans[:,0], ans[:,1], c=test_ys[:])
plt.savefig('siamese.png')
print("outdim=25")
# plt.show()

