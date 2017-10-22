#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib import rnn
from tensorflow.python.ops.math_ops import sigmoid
import inspect
import time
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops import nn_ops

import numpy as np
import tensorflow as tf
import collections
from sklearn.metrics import confusion_matrix

import pickle
import my_reader
import basic_lstmn4 as my_model
from tensorflow.contrib.rnn import LSTMStateTuple
flags = tf.flags
logging = tf.logging

matrix_dir ='/home/li/Desktop/TF/my/train_embedding&predict_punctuation3/'
embedding_save_dir = matrix_dir+"saved_model_data/embedding/embedding.ckpt"
biLSTM_model_save_dir = matrix_dir+"saved_model_data/model_biLSTM/model_biLSTM.ckpt"
##############################################################################
#flags.DEFINE_string(
#    "model", "small",
#    "A type of model. Possible options are: small, medium, large.")
#
#flags.DEFINE_string("data_path_2", '/home/li/Desktop/TF/my/icwb2_data/training/',
#                    "Where the training/test data is stored.")
#
#flags.DEFINE_string("test_data_path", '/home/li/Desktop/TF/my/icwb2_data/gold/',
#                    "Where the testing/test data is stored.")
#
#flags.DEFINE_string("save_path_2", 'save_model_all/',
#                    "Model output directory.")
#
#flags.DEFINE_bool("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")
##############################################################################


FLAGS = flags.FLAGS


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
#  num_steps = 20
  num_steps = 100
  
  num_states = 100
  
#  hidden_size = 200
  hidden_size = 200
  
#  max_epoch = 4
  max_epoch = 16
  
#  max_max_epoch = 13
  max_max_epoch = 50
  
#  keep_prob = 1.0
  keep_prob = 0.7
  
#  lr_decay = 0.5
  lr_decay = 0.9
  
#  batch_size = 20
  batch_size = 2
  
  vocab_size = 6000
  num_class=9*2+3


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
#  num_steps = 35
  num_steps = 100
  
  hidden_size = 650
#  max_epoch = 6
  max_epoch = 8
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.9
#  batch_size = 20
#  vocab_size = 10000

  batch_size = 30
  vocab_size = 6000
  num_class=9*2+3


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
#  num_steps = 35
  hidden_size = 1500
#  hidden_size = 1000
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
#  batch_size = 20
  batch_size = 20
  
#  vocab_size = 10000

  num_steps = 100
  vocab_size = 10000
  
  num_class=9*2+3


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
#############################################
def weight_variable(name,shape,dtype=tf.float32):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return vs.get_variable(name,initializer=initial,dtype=dtype)
    
def bias_variable(name,shape,dtype=tf.float32):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return vs.get_variable(name,initializer=initial,dtype=dtype)

def gate_conv2(name,_input,kernel_shanpe=[1,3,3,1],strides=[1,1,1,1],padding='SAME',dtype=tf.float32):
    w_conv_raw=weight_variable(name+"w_raw",kernel_shanpe,dtype=dtype)
    b_conv_raw=bias_variable(name+"b_raw",[kernel_shanpe[3]],dtype=dtype)
    
    res_conv_raw=tf.nn.conv2d(_input, w_conv_raw,strides,padding=padding)
    res_raw=res_conv_raw+b_conv_raw
    
    w_conv_act=weight_variable(name+"w_act",kernel_shanpe,dtype=dtype)
    b_conv_act=bias_variable(name+"b_act",[kernel_shanpe[3]],dtype=dtype)
    
    res_conv_act=tf.nn.conv2d(_input, w_conv_act,strides,padding=padding)
    res_act=sigmoid(res_conv_act+b_conv_act)
    
    res=res_raw * res_act
    return res

def normal_conv2(name,_input,kernel_shanpe=[1,3,3,1],strides=[1,1,1,1],padding='SAME',dtype=tf.float32):
    
    w_conv=weight_variable(name+"w_",kernel_shanpe,dtype=dtype)
    b_conv=bias_variable(name+"b_",[kernel_shanpe[3]],dtype=dtype)
    
    res_conv=tf.nn.conv2d(_input, w_conv,strides,padding=padding)
    res=res_conv+b_conv
    
    return res


class my_DropoutWrapper(DropoutWrapper):
    def __init__(self, cell, keep_prob=1.0,seed=None):
        """Create a cell with added input and/or output dropout.
    
        Dropout is never used on the state.
    
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.
    
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        if (isinstance(keep_prob, float) and
            not (keep_prob >= 0.0 and keep_prob <= 1.0)):
          raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                           % keep_prob)
        self._cell = cell
        self._keep_prob = keep_prob
        self._seed = seed
    
    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._keep_prob, float) or
            self._keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._keep_prob, seed=self._seed)
            c,h=state
            h = nn_ops.dropout(h, self._keep_prob, seed=self._seed)
            state=LSTMStateTuple(c,h)
        output, new_state = self._cell(inputs, state, scope)
        return output, new_state




def tensor_filter(input_x,element_before,element_after):
    if input_x.dtype !=tf.int32:
        input_x=tf.cast(input_x,tf.int32)
        
    shape=input_x.get_shape()
    canvas=tf.zeros(shape)
    for i in element_before:
        keynote=tf.fill(shape,i)
        brush=tf.equal(keynote,input_x)
        brush=tf.cast(brush, tf.float32)
        canvas=canvas+brush
    input_x=tf.cast(input_x, tf.float32)
    sidekick=(tf.ones(shape)-canvas)*input_x
    protagonist=canvas*element_after
    masterpiece=protagonist + sidekick
    masterpiece=tf.cast(masterpiece,tf.int32)
    return masterpiece
def joint_tensor_filter(input_x,element_tape_before,element_tape_after):
    data=input_x
    for i in range(len(element_tape_before)):
        data=tensor_filter(data,element_tape_before[i],element_tape_after[i])
    return data

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
#    self.epoch_size = ((len(data[0]) // batch_size) - 1) // num_steps
#    self.epoch_size = ((len(data[0]) // batch_size)+1 ) // num_steps
    self.input_data, self.targets,self.epoch_size = my_reader.ptb_producer_IO(
        data, batch_size, num_steps,interval_size=num_steps, name=name,
        index_bound={'#':num_steps,'；':num_steps,'。':num_steps,'？':num_steps,'！':num_steps,'，':num_steps,' ':num_steps})
    self.num_class=config.num_class
    print('xxxxxxxxxxxxxxxxxxxxxxxxxx,,,epoch_size:...................\n',self.epoch_size,'\n................end')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxx,,,len(data):...................\n',len(data),'\n................end')
    
    
    
    print('xxxxxxxxxxxxxxxxxxxxxxxxxx,,,iniput_data:...................\n',self.input_data,'\n................end')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxx,,,targets:...................\n',self.input_data,'\n................end')
#    print('reader.ptb_producer.............\n',self.input_data,'\n',self.targets,'\n...........end')




class PTBModel(object):
  """The PTB model."""
#####################################################################################################################################
  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    num_class=config.num_class

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    
    
    def lstmn_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
#     ###############################################################################     
#      if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
#        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
#      else:
#        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    
#      if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
#        return my_model.BasicLSTMNcell(size, num_states=30,forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
#      else:
#        return my_model.BasicLSTMNcell(size, num_states=30,forget_bias=0.0, state_is_tuple=True)
    

      if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
#        return my_model.BasicLSTMNcell(size, num_states=config.num_states,forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
#      else:
#        return my_model.BasicLSTMNcell(size, num_states=config.num_states,forget_bias=0.0, state_is_tuple=True)
    
        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    
    def output_layer_lstm():
        
      if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
#        return my_model.BasicLSTMNcell(size, num_states=config.num_states,forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
#      else:
#        return my_model.BasicLSTMNcell(size, num_states=config.num_states,forget_bias=0.0, state_is_tuple=True)
    
        return tf.contrib.rnn.BasicLSTMCell(size*2, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(size*2, forget_bias=0.0, state_is_tuple=True)
    
    
    ###############################################################################

    
    attn_cell = lstmn_cell
    first_layer_cell = lstmn_cell
    attn_output_layer_lstm=output_layer_lstm
#    print(attn_cell)
    
    
    if is_training and config.keep_prob < 1:
      def attn_cell():
          return tf.contrib.rnn.DropoutWrapper(lstmn_cell(), output_keep_prob=config.keep_prob)
#        return my_DropoutWrapper(lstmn_cell(), keep_prob=config.keep_prob)
#        return my_model.DropoutWrapper(lstmn_cell(), output_keep_prob=config.keep_prob)
#        return my_model.DropoutWrapper_ch_previous(lstmn_cell(), num_states=config.num_states,output_keep_prob=config.keep_prob)
      def my_attn_cell():
          return tf.contrib.rnn.DropoutWrapper(lstmn_cell(), input_keep_prob=config.keep_prob ,output_keep_prob=config.keep_prob)

      def last_layer_cell():
          return tf.contrib.rnn.DropoutWrapper(lstmn_cell(),input_keep_prob=config.keep_prob,
                                               output_keep_prob=config.keep_prob)
          
      def first_layer_cell():
          return tf.contrib.rnn.DropoutWrapper(lstmn_cell(),input_keep_prob=config.keep_prob)
    
    
      def attn_output_layer_lstm():
          return tf.contrib.rnn.DropoutWrapper(output_layer_lstm(),output_keep_prob=config.keep_prob)
    
    
    
#    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
#    self._initial_state = cell.zero_state(batch_size, data_type())
    self._initial_state = None
    
    
    
    

    with tf.device("/gpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], trainable=False,dtype=data_type())
#      embedding = tf.Variable([vocab_size, size], dtype=data_type(),name="embedding")
      
      print('embedding:...................\n',embedding,'\n................end')      
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      print('inputs(before dropout):...................\n',inputs,'\n................end')
       
    self._embedding=embedding
      
#    if is_training and config.keep_prob < 1:
#      inputs = tf.nn.dropout(inputs, config.keep_prob) #for preventing overritting
      
    inputs=tf.unstack(inputs, num_steps, 1)
    print('inputs(after dropout):...................\n',inputs,'\n................end')
    original_inputs=tf.reshape(tf.concat(inputs,axis=1),[-1,size])
     
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    state = self._initial_state
    
#    with tf.variable_scope("RNN"):
#      for time_step in range(num_steps):
#        if time_step > 0: tf.get_variable_scope().reuse_variables()
#        (cell_output, state) = cell(inputs[:, time_step, :], state)
#        outputs.append(cell_output)
##        print('outputs:...................\n',outputs,'\n................end')

#    lstmn_fw_cell = attn_cell()
#    # Backward direction cell
#    lstmn_bw_cell = attn_cell()
    
#    lstmn_fw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
#    lstmn_bw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    cell_head = tf.contrib.rnn.MultiRNNCell([first_layer_cell(),first_layer_cell()], state_is_tuple=True)
    cell_follow= tf.contrib.rnn.MultiRNNCell([output_layer_lstm() ,output_layer_lstm()], state_is_tuple=True)
#    cell= tf.contrib.rnn.MultiRNNCell([first_layer_cell() ,first_layer_cell()], state_is_tuple=True)
    



    cell_tape=[cell_head,first_layer_cell(),first_layer_cell(),first_layer_cell(),first_layer_cell()]
    cell_num_layers=len(cell_tape)
    with vs.variable_scope("RNN"):
        for i in range(cell_num_layers):
            with vs.variable_scope("RNN_%d" % i) as RNNscope:
            
                outputs, _, _ = rnn.static_bidirectional_rnn(cell_tape[i], cell_tape[i], inputs,scope=RNNscope,dtype=data_type())
                
            value=tf.concat(outputs,axis=1)
            outputs=tf.split(value=value,num_or_size_splits=num_steps*2,axis=1)
            temp_x=tf.concat(outputs[::2],axis=1)
            temp_y=tf.concat(outputs[1::2],axis=1)
            outputs=temp_x+temp_y
            if i<cell_num_layers-1:
                outputs=tf.split(value=outputs, num_or_size_splits=num_steps, axis=1)
                inputs=outputs
                
#            conv_input = tf.reshape(tf.concat(values=outputs,axis=1), [batch_size,1,num_steps*size*2,1])
#            with vs.variable_scope("conv_scope") as _conv_scope:
##                if i > 0: _conv_scope.reuse_variables()
#                outputs=normal_conv2("output_conv_%d"%i,conv_input ,kernel_shanpe=[1, size*2 , 1, size],
#                                     strides=[1,1,size*2,1],padding='VALID',dtype=tf.float32)
#            outputs=tf.reshape(outputs,[batch_size,-1])
#            if i<config.num_layers-1:
#                outputs=tf.split(value=outputs, num_or_size_splits=num_steps, axis=1)
#                inputs=outputs
       
       
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
#    output=tf.concat([original_inputs,output],axis=1)

    #################
#    softmax_w1 = tf.get_variable( "softmax_w1", [size*2, size*2],  dtype=data_type() )
#    softmax_b1 = tf.get_variable("softmax_b1", [size*2], dtype=data_type())
#
#    output=sigmoid(tf.matmul(output,softmax_w1)+softmax_b1)
    #################

    softmax_w = tf.get_variable( "softmax_w", [size, num_class],  dtype=data_type() )
    softmax_b = tf.get_variable("softmax_b", [num_class], dtype=data_type())
    
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    
    
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

##########################################################################################################################

    output_label_CWS_PU=tf.argmax(logits,1)
    output_label_CWS_PU=tf.reshape(output_label_CWS_PU,[batch_size,-1])
    output_label_CWS_PU=tf.cast(output_label_CWS_PU,tf.int32)
    ######################################################################################################################
    #compute the label of CWS                                                                                           ##
    label_single_character=list(range(3,12))                                                                            ##
    label_right_character=list(range(12,21))                                                                            ##
    targets_label_CWS=joint_tensor_filter(input_.targets, [label_single_character,label_right_character] ,[3,4])        ##
    output_label_CWS=joint_tensor_filter(output_label_CWS_PU, [label_single_character,label_right_character] ,[3,4])    ##
                                                                                                                        ##
    # compute the label of PU                                                                                           ##
    transformer_2PU_X=[[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]]          ##
    transformer_2PU_Y=[ 1 , 2 , 3,  4,  5,  6,  7,  8,  9,   1,   2,   3,   4,   5,   6,   7,   8,   9,   1 ]           ##
    targets_label_PU=joint_tensor_filter(input_.targets,transformer_2PU_X ,transformer_2PU_Y)                           ##
    output_label_PU=joint_tensor_filter(output_label_CWS_PU,transformer_2PU_X ,transformer_2PU_Y)                       ##
    ######################################################################################################################
    self._true_CWS_PU=tf.reshape(input_.targets,[-1])
    self._pred_CWS_PU=tf.reshape(output_label_CWS_PU,[-1])
    
    self._true_CWS=tf.reshape(targets_label_CWS,[-1])
    self._pred_CWS=tf.reshape(output_label_CWS,[-1])
    
    self._true_PU=tf.reshape(targets_label_PU,[-1])
    self._pred_PU=tf.reshape(output_label_PU,[-1])
    

#   compute the accuracy of cws&pu
    correct_predict_CWS_PU = tf.equal(output_label_CWS_PU,input_.targets )
    accuracy_predict_CWS_PU = tf.reduce_mean(tf.cast(correct_predict_CWS_PU, tf.float32))
#   compute the accuracy of cws
    correct_predict_CWS = tf.equal(output_label_CWS,targets_label_CWS)
    accuracy_predict_CWS = tf.reduce_mean(tf.cast(correct_predict_CWS, tf.float32))
    
    correct_predict_PU = tf.equal(output_label_PU,targets_label_PU)
    accuracy_predict_PU = tf.reduce_mean(tf.cast(correct_predict_PU, tf.float32))
    
    self._accuracy_predict_CWS_PU=accuracy_predict_CWS_PU
    self._accuracy_predict_CWS=accuracy_predict_CWS
    self._accuracy_predict_PU=accuracy_predict_PU
    
###################################################################################################################################
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    
    tvars = tf.trainable_variables()
    print('trainable_variables:...................\n',tvars,'\n................end')
    
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
    #clip the gradient to prevent gradient exploding or vanish
    
    print('grads:...................\n',grads,'\n................end')
    
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step()   )
    
#    optimizer = tf.train.GradientDescentOptimizer(self._lr)
#    self._train_op = optimizer.minimize(cost)
#################################################################################################

  @property
  def accuracy(self):
    return self._accuracy_predict_CWS_PU ,  self._accuracy_predict_CWS , self._accuracy_predict_PU

  @property
  def jointly_T_P(self):
    return self._true_CWS_PU  ,  self._pred_CWS_PU

  @property
  def CWS_T_P(self):
    return self._true_CWS ,  self._pred_CWS

  @property
  def PU_T_P(self):
    return self._true_PU ,  self._pred_PU


############################################################################

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


  def load_embedding(self,session,saver):
      saver.restore(session,embedding_save_dir)



  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def embedding(self):
    return self._embedding


def compute_R_P(y_true, y_pred,num_class):
    if len(y_pred)!=len(y_true):
        raise ValueError("len(y_true) and len(y_pred) must be equal")
    
    num_example=len(y_true)
    
    dict_class=dict.fromkeys(range(num_class),0)   #build the ariginal dict_class 
    real_dict_class=dict(collections.Counter(y_true)) 
    dict_class.update(real_dict_class) 
    
    dict_class_portion={}
    for i in range(num_class):
        dict_class_portion[i]=dict_class[i]/num_example
        
        
    dict_recall={}
    dict_precision={}
    class_array=list(range(num_class))
    matrix_confusion=confusion_matrix(y_true,y_pred,labels=class_array)
    for i in range(num_class):
        dict_recall[i]=matrix_confusion[i,i]/sum(matrix_confusion[:,i]) if matrix_confusion[i,i]!=0 else 0
        dict_precision[i]=matrix_confusion[i,i]/sum(matrix_confusion[i,:]) if matrix_confusion[i,i]!=0 else 0
    
        
#    total_recall=tf.convert_to_tensor(list(dict_recall.values())) * tf.convert_to_tensor(list(dict_class_portion.values()))
#    total_recall=tf.reduce_sum(total_recall)/num_class
#
#    total_precision=tf.convert_to_tensor(list(dict_precision.values()))*tf.convert_to_tensor(list(dict_class_portion.values()))
#    total_precision=tf.reduce_sum(total_precision)/num_class

#    total_recall=np.array(list(dict_recall.values())) * np.array(list(dict_class_portion.values()))
#    total_recall=sum(total_recall)
#    total_precision=np.array(list(dict_precision.values())) * np.array(list(dict_class_portion.values()))
#    total_precision=sum(total_precision)
    
    total_recall=np.array(list(dict_recall.values()))
    total_recall=np.mean(total_recall)
    total_precision=np.array(list(dict_precision.values())) 
    total_precision=np.mean(total_precision)
    
    
    
    
    F1=2*total_precision*total_recall/(total_precision+total_recall)
    
    return total_precision,total_recall,F1


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  
  start_time = time.time()
  costs = 0.0
  iters = 0
  whole_accuracy_CWS_PU=0
  whole_accuracy_CWS=0
  whole_accuracy_PU=0
  
  whole_precision_CWS_PU=0
  whole_recall_CWS_PU=0
  whole_F1_CWS_PU=0
  
  whole_precision_CWS=0
  whole_recall_CWS=0
  whole_F1_CWS=0
  
  whole_precision_PU=0
  whole_recall_PU=0
  whole_F1_PU=0

#  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
#      "final_state": model.final_state,
      "accuracy":model.accuracy,
        "CWS_PU_T_P":model.jointly_T_P,
        "CWS_T_P":model.CWS_T_P,
        "PU_T_P":model.PU_T_P
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
    
  epoch_size=model.input.epoch_size
  for step in range(epoch_size):
#    print('this is a test...............................test......test')
#    feed_dict = {}
#    for i, (c, h) in enumerate(model.initial_state):
#      feed_dict[c] = state[i].c
#      feed_dict[h] = state[i].h
#      print('state:.................',state,'\n...........en')
#      print('feed_dict:.................',feed_dict,'\n...........en')


#    vals = session.run(fetches, feed_dict)
    vals = session.run(fetches)
#    cost =session.run(model.cost)
#    session.run(eval_op)
    cost = vals["cost"]
#    state = vals["final_state"]
    accuracy_CWS_PU , accuracy_CWS , accuracy_PU=vals["accuracy"]
    target_CWS_PU , predict_CWS_PU=vals["CWS_PU_T_P"]
    target_CWS , predict_CWS=vals["CWS_T_P"]
    target_PU  , predict_PU=vals["PU_T_P"]

    def means_of(matrix ,CP):
        if matrix != 0:
            return (matrix+CP)/2
        else:
            return CP
        
        
    costs += cost
    iters += model.input.num_steps
    
#    whole_accuracy_CWS_PU =means_of(whole_accuracy_CWS_PU,accuracy_CWS_PU)#accuracy of CWS and PU
#    whole_accuracy_CWS =means_of(whole_accuracy_CWS,accuracy_CWS)#accuracy of CWS
#    whole_accuracy_PU =means_of(whole_accuracy_PU,accuracy_PU)#accuracy of PU
    
    whole_accuracy_CWS_PU =whole_accuracy_CWS_PU+accuracy_CWS_PU
    whole_accuracy_CWS =whole_accuracy_CWS+accuracy_CWS
    whole_accuracy_PU =whole_accuracy_PU+accuracy_PU
    
    precision_CWS_PU , recall_CWS_PU , F1_CWS_PU = compute_R_P(target_CWS_PU,predict_CWS_PU,21)
    precision_CWS , recall_CWS , F1_CWS = compute_R_P(target_CWS,predict_CWS,5)
    precision_PU , recall_PU , F1_PU = compute_R_P(target_PU,predict_PU,10)
    
#    whole_precision_CWS_PU=means_of(whole_precision_CWS_PU,precision_CWS_PU)
#    whole_recall_CWS_PU=means_of(whole_recall_CWS_PU,recall_CWS_PU)
#    whole_F1_CWS_PU=means_of(whole_F1_CWS_PU,F1_CWS_PU)
    
    whole_precision_CWS_PU=whole_precision_CWS_PU + precision_CWS_PU
    whole_recall_CWS_PU=whole_recall_CWS_PU + recall_CWS_PU
    whole_F1_CWS_PU=whole_F1_CWS_PU + F1_CWS_PU
    
#    whole_precision_CWS=means_of(whole_precision_CWS,precision_CWS)
#    whole_recall_CWS=means_of(whole_recall_CWS,recall_CWS)
#    whole_F1_CWS=means_of(whole_F1_CWS,F1_CWS)

    whole_precision_CWS=whole_precision_CWS + precision_CWS
    whole_recall_CWS=whole_recall_CWS + recall_CWS
    whole_F1_CWS=whole_F1_CWS + F1_CWS
    
#    whole_precision_PU=means_of(whole_precision_PU,precision_PU)
#    whole_recall_PU=means_of(whole_recall_PU,recall_PU)
#    whole_F1_PU=means_of(whole_F1_PU,F1_PU)

    whole_precision_PU=whole_precision_PU + precision_PU
    whole_recall_PU=whole_recall_PU + recall_PU
    whole_F1_PU=whole_F1_PU + F1_PU
    
    if verbose and step % (epoch_size // 10) == 10:
      print("      %.3f     perplexity: %.3f     speed: %.0f wps" %
            ( step * 1.0 / epoch_size,
             np.exp(costs/iters),
             iters * model.input.batch_size / (time.time() - start_time)
             ))
      print("           ||   accuracy  ||   precision  ||   recall  ||     F1    ||")
      print("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     ||"\
            %(accuracy_CWS_PU,precision_CWS_PU,recall_CWS_PU,F1_CWS_PU))
      
      print("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     ||"\
            %(accuracy_CWS,precision_CWS,recall_CWS,F1_CWS))
      
      print("    PU     || %.3f       || %.3f        || %.3f     || %.3f     ||"\
            %(accuracy_PU,precision_PU,recall_CWS,F1_PU))
      
      
#      with open("result.txt","a+") as f:
#          f.write("      %.3f     perplexity: %.3f     speed: %.0f wps \n" %
#                ( step * 1.0 / model.input.epoch_size,
#                 np.exp(costs/iters),
#                 iters * model.input.batch_size / (time.time() - start_time)
#                 ))
#          f.write("           ||   accuracy  ||   precision  ||   recall  ||     F1    || \n")
#          f.write("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     || \n"\
#                %(accuracy_CWS_PU,precision_CWS_PU,recall_CWS_PU,F1_CWS_PU))
#          
#          f.write("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     || \n"\
#                %(accuracy_CWS,precision_CWS,recall_CWS,F1_CWS))
#          
#          f.write("    PU     || %.3f       || %.3f        || %.3f     || %.3f     || \n"\
#                %(accuracy_PU,precision_PU,recall_CWS,F1_PU))
      
  information_CWS_PU=whole_accuracy_CWS_PU/epoch_size, whole_precision_CWS_PU/epoch_size , \
                      whole_recall_CWS_PU/epoch_size , whole_F1_CWS_PU/epoch_size
                      
  information_CWS=whole_accuracy_CWS/epoch_size , whole_precision_CWS/epoch_size , \
                      whole_recall_CWS/epoch_size, whole_F1_CWS/epoch_size
                      
  information_PU=whole_accuracy_PU/epoch_size, whole_precision_PU/epoch_size , \
                  whole_recall_PU/epoch_size , whole_F1_PU/epoch_size


  return np.exp(costs/iters),information_CWS_PU,information_CWS,information_PU


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)




def main(_):
  if not FLAGS.data_path_2:
    raise ValueError("Must set --data_path_2 to PTB data directory")


  with open(matrix_dir+'original_saved_model_data/word_dict/vocabulary.pickle', 'rb') as f:#load the saved dict 
    dict_data = pickle.load(f)
    
  raw_data = my_reader.ptb_raw_data_with_saved_dict(FLAGS.data_path_2, data_path_test=FLAGS.test_data_path,dict_data=dict_data)
  train_data, valid_data, test_data= raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("test"):
      train_input = PTBInput(config=config, data=test_data, name="Test_Input")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=False, config=config, input_=train_input)
#        print('m.initial_state...............', m.initial_state)
        
#      tf.summary.scalar("Training_Loss", m.cost)
#      tf.summary.scalar("Learning_Rate", m.lr)
      
#      tf.summary.scalar("Learning_Rate", m.accuracy)
#      tf.summary.scalar("jointly_T_P", m.jointly_T_P)
#      tf.summary.scalar("CWS_T_P", m.CWS_T_P)
#      tf.summary.scalar("PU_T_P", m.PU_T_P)

      
#      tf.summary.scalar("accuracy_PP", m.accuracy_PP)

#    with tf.name_scope("Valid"):
#      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
#      with tf.variable_scope("Model", reuse=True, initializer=initializer):
#        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
#      tf.summary.scalar("Validation Loss", mvalid.cost)

#    with tf.name_scope("Test"):
#      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
#      with tf.variable_scope("Model", reuse=True, initializer=initializer):
#        mtest = PTBModel(is_training=False, config=eval_config,input_=test_input)
        
        
    saver_of_embedding = tf.train.Saver({"my_embedding": m.embedding})
#    saver_of_model_biLSTM = tf.train.Saver(tf.global_variables())
    saver_of_model_biLSTM = tf.train.Saver()
    
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path_2)
    with sv.managed_session() as session:
#      m.load_embedding(session,saver_of_embedding)
      
#      for i in range(config.max_max_epoch):
#        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
#        m.assign_lr(session, config.learning_rate * lr_decay)
#
#        print("Epoch: %d   Learning rate: %.3f" % (i + 1, session.run(m.lr)))
#        
#        train_perplexity,information_CWS_PU,information_CWS ,information_PU\
#        = run_epoch(session, m, eval_op=m.train_op, verbose=True)
#    
#        print("the total information\n")
#        print("Train_Perplexity: %.3f"% (train_perplexity))
#        print("           ||  accuracy   ||   precision  ||   recall  ||     F1    ||")
#        print("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     ||" %information_CWS_PU)
#        print("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     ||" %information_CWS)
#        print("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     ||" %information_PU)
#        print("\n")
#        
#        with open("train_result.txt","a+") as f:
#            f.write("the total information \n")
#            f.write("Train_Perplexity: %.3f \n"% (train_perplexity))
#            f.write("           ||  accuracy   ||   precision  ||   recall  ||     F1    || \n")
#            f.write("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     || \n" %information_CWS_PU)
#            f.write("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     || \n" %information_CWS)
#            f.write("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     || \n" %information_PU)
#            f.write("\n")

#        valid_perplexity = run_epoch(session, mvalid)
#        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        

#      test_perplexity = run_epoch(session, mtest)
      test_perplexity,test_information_CWS_PU,test_information_CWS ,test_information_PU=run_epoch(session, m,verbose=True)
      print("the total test_information\n")
      print("Train_Perplexity: %.3f"% (test_perplexity))
      print("           ||  accuracy   ||   precision  ||   recall  ||     F1    ||")
      print("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     ||" %test_information_CWS_PU)
      print("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     ||" %test_information_CWS)
      print("    PU     || %.3f       || %.3f        || %.3f     || %.3f     ||" %test_information_PU)
      print("\n")
      
      with open("test_result.txt","a+") as f:
            f.write("the total information \n")
            f.write("Train_Perplexity: %.3f \n"% (test_perplexity))
            f.write("           ||  accuracy   ||   precision  ||   recall  ||     F1    || \n")
            f.write("  CWS_PU   || %.3f       || %.3f        || %.3f     || %.3f     || \n" %test_information_CWS_PU)
            f.write("    CWS    || %.3f       || %.3f        || %.3f     || %.3f     || \n" %test_information_CWS)
            f.write("    PU    || %.3f       || %.3f        || %.3f     || %.3f     || \n" %test_information_PU)
            f.write("\n")



if __name__ == "__main__":
  tf.app.run()






























