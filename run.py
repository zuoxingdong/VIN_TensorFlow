import time
import argparse

import numpy as np
import tensorflow as tf

from model import *
from dataset import *

def train_or_eval(dataset, args, feed_ops, eval_ops):
    num_batches = dataset.num_examples//args.batch_size
    total_examples = num_batches*args.batch_size
    
    assert len(eval_ops) == 2 or len(eval_ops) == 3
    if len(eval_ops) == 3: # [train_step, num_err, loss]
        train_mode = True
    else: # test mode: [num_err, loss]
        train_mode = False

    total_err = 0.0
    total_loss = 0.0
    
    for batch in range(num_batches):
        X, S1, S2, y = feed_ops
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(args.batch_size)
        
        feed_dict = {X: X_batch,
                     S1: S1_batch, 
                     S2: S2_batch, 
                     y: y_batch}
        
        if train_mode:
            _, err, loss = sess.run(eval_ops, feed_dict)
        else:
            err, loss = sess.run(eval_ops, feed_dict)
            
        total_err += err
        total_loss += loss
        
    return total_err/total_examples, total_loss/total_examples


# Parsing training parameters
parser = argparse.ArgumentParser()

parser.add_argument('--datafile', 
                    type=str, 
                    default='../data/gridworld_8x8.npz', 
                    help='Path to data file')
parser.add_argument('--imsize', 
                    type=int, 
                    default=8, 
                    help='Size of image')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.002, 
                    help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
parser.add_argument('--epochs', 
                    type=int, 
                    default=30, 
                    help='Number of epochs to train')
parser.add_argument('--k', 
                    type=int, 
                    default=10, 
                    help='Number of Value Iterations')
parser.add_argument('--ch_i', 
                    type=int, 
                    default=2, 
                    help='Number of channels in input layer')
parser.add_argument('--ch_h', 
                    type=int, 
                    default=150, 
                    help='Number of channels in first hidden layer')
parser.add_argument('--ch_q', 
                    type=int, 
                    default=10, 
                    help='Number of channels in q layer (~actions) in VI-module')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=128, 
                    help='Batch size')
parser.add_argument('--use_log', 
                    type=bool, 
                    default=False, 
                    help='True to enable TensorBoard summary')
parser.add_argument('--logdir', 
                    type=str, 
                    default='.log/', 
                    help='Directory to store TensorBoard summary')

args = parser.parse_args()

# Define placeholders

# Input tensor: Stack obstacle image and goal image, i.e. ch_i = 2
X = tf.placeholder(tf.float32, shape=[None, args.imsize, args.imsize, args.ch_i], name='X')
# Input batches of vertical positions
S1 = tf.placeholder(tf.int32, shape=[None], name='S1')
# Input batches of horizontal positions
S2 = tf.placeholder(tf.int32, shape=[None], name='S2')
# Labels: actions {0,...,7}
y = tf.placeholder(tf.int64, shape=[None], name='y')

# VIN model
logits, prob_actions = VIN(X, S1, S2, args)

# Loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean') 
#######################################################################################
########### ASK QUESTIONS: ONLY PRINT EPOCH LOSS, reduce_sum or reduce_mean  ##########
#######################################################################################

# Optimizer
optimizer = tf.train.RMSPropOptimizer(args.lr, epsilon=1e-6, centered=True)

# Train op
train_step = optimizer.minimize(loss)

# Select actions wit max probability 
actions = tf.argmax(prob_actions, 1)

# Number of wrongly selected actions
num_err = tf.reduce_sum(tf.to_float(tf.not_equal(actions, y)))
#######################################################################################
########### ASK QUESTIONS: ONLY PRINT EPOCH LOSS, reduce_sum or reduce_mean  ##########
#######################################################################################

# Initialization of variables
init_op = tf.global_variables_initializer()

# Load the dataset
trainset = Dataset(args.datafile, mode='train', imsize=args.imsize)
testset = Dataset(args.datafile, mode='test', imsize=args.imsize)

# Running
with tf.Session() as sess:
    
    # Intialize all variables
    sess.run(init_op)
    
    for epoch in range(args.epochs): # Each epoch iterates over whole dataset
        start_time = time.time() # Time duration for current epoch
        
        # Train for one step and evaluate error rate and mean loss
        mean_err, mean_loss = train_or_eval(trainset, 
                                            args,
                                            feed_ops=[X, S1, S2, y], 
                                            eval_ops=[train_step, num_err, loss])
        
        # Print logs per epoch
        time_duration = time.time() - start_time
        out_str = 'Epoch: {:3d} ({:.1f} s): \n\t Train Loss: {:.5f} \t Train Err: {:.5f}'
        print(out_str.format(epoch, time_duration, mean_loss, mean_err))
    print('\n Finished training...\n ')
    
    # Testing
    print('\n Testing...\n')
    
    mean_err, mean_loss = train_or_eval(testset, args, feed_ops=[X, S1, S2, y], eval_ops=[num_err, loss])
    print('Test Accuracy: {:.2f}%'.format(100*(1 - mean_err)))
    
    # Reward and value images
    
    # Process test set
    Xtest = testset.images
    S1test = testset.s1
    S2test = testset.s2
    ytest = testset.labels
    
    # Collection of reward and value images
    r = tf.get_collection('r')
    v = tf.get_collection('v')
    
    idx = np.random.choice(testset.num_examples, size=10, replace=False)
    r_arr, v_arr = sess.run([r, v], feed_dict={X: Xtest[idx]})
    np.savez_compressed('reward_value_images', [Xtest[idx], r_arr, v_arr])
