#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 20:26:02 2017

@author: li
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import collections
import os
import pickle

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS
matrix_dir ='/home/li/Desktop/TF/my/train_embedding&predict_punctuation2/'

with open(matrix_dir+'original_saved_model_data/word_dict/vocabulary.pickle', 'rb') as f:#load the saved dict 
    dict_data = pickle.load(f)
#####################################################################################
##please blocking this part then you run the code in second times
#flags.DEFINE_string(
#    "training_dir", "icwb2_data/training/pku_training.utf8",
#    "the addrss of training data")
#
#flags.DEFINE_string(
#    "testing_dir", "icwb2_data/training/pku_training.utf8",
#    "the addrss of test data")
#
#flags.DEFINE_string(
#    "row_test_dir", "row_test.utf8",
#    "a long way to go,this is a test file to test the function of reader")
#flags.DEFINE_string("data_path", '/home/li/Desktop/TF/my/icwb2_data/training/',
#                    "Where the training/test data is stored.")
##
###################################################################################
def B2Q(uchar):
        """半角转全角"""
        inside_code=ord(uchar)
        if inside_code<0x0020 or inside_code>0x7e:      #不是半角字符就返回原来的字符
                return uchar
        if inside_code==0x0020: #除了空格其他的全角半角的公式为:半角=全角-0xfee0
                inside_code=0x3000
        else:
                inside_code+=0xfee0
        return chr(inside_code)
    
    
def Q2B(uchar):
        """全角转半角"""
        inside_code=ord(uchar)
        if inside_code==0x3000:
                inside_code=0x0020
        else:
                inside_code-=0xfee0
        if inside_code<0x0020 or inside_code>0x7e:      #转完之后不是半角字符返回原来的字符
                return uchar
        return chr(inside_code)



def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
        else:
                return False
            
def only_chinese(str):
    """ str must be a string list"""
    new_str=[]
    for i in str:
        if is_chinese(i):
            new_str.append(i)
    return new_str         
   
def filtering_chinese(str):
    new_str=[]
    for i in str:
        if not is_chinese(i):
            new_str.append(i)
    return new_str


 
def is_number(uchar):
        """判断一个unicode是否是数字"""
        if (uchar >= u'\u0030' and uchar<=u'\u0039') or (Q2B(uchar) >= u'\u0030' and Q2B(uchar)<=u'\u0039'):
                return True
        else:
                return False
def only_number(str):
    new_str=[]
    for i in str:
        if is_number(i):
            new_str.append(i)
    return new_str
def filtering_number(str):
    new_str=[]
    for i in str:
        if not is_number(i):
            new_str.append(i)
    return new_str                  




def is_alphabet(uchar):
        """判断一个unicode是否是英文字母"""
        if ((uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a')) or \
            ((Q2B(uchar) >= u'\u0041' and Q2B(uchar)<=u'\u005a') or (Q2B(uchar) >= u'\u0061' and Q2B(uchar)<=u'\u007a')):
                return True
        else:
                return False
def only_alphabet(str):
    new_str=[]
    for i in str:
        if is_alphabet(i):
            new_str.append(i)
    return new_str

def filtering_alphabet(str):
    new_str=[]
    for i in str:
        if not is_alphabet(i):
            new_str.append(i)
    return new_str
            
            
 
def is_other(uchar):
        """判断是否非汉字，数字和英文字符"""
        if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
                return True
        else:
                return False
            
def filtering_other(str):
    """ the input is a string,
    and this runction's purpose is to filtering the other character which is not chinese number or alphabet """
    new_str=[]
    for i in str:
        if not is_other(i):
            new_str.append(i)
    return new_str           

            

 
def stringQ2B(ustring):
        """把字符串全角转半角"""
        return "".join([Q2B(uchar) for uchar in ustring])
 
def uniform(ustring):
        """格式化字符串，完成全角转半角，大写转小写的工作"""
        return stringQ2B(ustring).lower()
 
def string2List(ustring):
        """将ustring按照中文，字母，数字分开"""
        retList=[]
        utmp=[]
        for uchar in ustring:
                if is_other(uchar):
                        if len(utmp)==0:
                                continue
                        else:
                                retList.append("".join(utmp))
                                utmp=[]
                else:
                        utmp.append(uchar)
        if len(utmp)!=0:
                retList.append("".join(utmp))
        return retList

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def only_chinese_except(str,*EXC):
    """ str must be a string list"""
    
    exc=[Q2B(i) for i in EXC]
    print(EXC)
    new_str=[]
    for i in str:
        if (i >= u'\u4e00' and i<=u'\u9fa5'):
            new_str.append(i)
        else:
            if (i in exc):new_str.append(i)
    return new_str

    
################################################################################################


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        data=f.read().decode("utf-8").replace("\n", '#')
        data=data.replace("\r\n",'#')
    final_data=[]
    for i in data:
        final_data.append(i)
#        if is_chinese(i):
#            final_data.append(i)
#        else:
#            final_data.append(Q2B(i))
                
    return final_data



def build_vocab(filename):
  data = read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id




def build_chinese_vocab_except_(filename,*mark):
  data = read_words(filename)
  new_data=[]
  for i in data:
      if i >= u'\u4e00' and i<=u'\u9fa5' or (i in mark):
          new_data.append(i)
  counter = collections.Counter(new_data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def build_vocab_excepte_(filename,*mark):
    """ other means the characters except chinese number and alphabet   """
    data = read_words(filename)
    for func in mark:
        
        data=func(data)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def convert2dict(data):
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id
    


def file_to_word_ids(filename, word_to_id):
  data = read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def convert_words_to_ids(data,word_to_id):
    return [word_to_id[word] for word in data if word in word_to_id]


def hon(x,y):
    for i in x:
        if i in y:
            return True
    return False

###############################################################################################
special_chinese=['-',   '°',   '±',  '·',     '×',   '—',   '…',   '‰',
                 
                 '℃',  'Ⅱ',  'Ⅲ',  '∶',    '①',  '②',  '③',   '④', 
                 
                 '⑤',  '⑥',  '⑦',  '⑵',   '▲',   '△',   '○',   '●',  
                 
                 '％',  '＋',  '－',  '．',   '／',  '＝',  '＞',   '～']
special_chinese=list(set(special_chinese))


bound9=['，',     '。',      '？',      '！',    '、',      '：',      '；',    '#',   ' ']



lvchu=[']',      '‘',        '’',      '“',     '”',       '〈',      '〉',     '《',    '》',
      
      '『',       '』',        '（',     '）',     '＊',      '［',     '］']
lvchu=list(set(lvchu))

################################################################################################
def readword_with_filter_nois_word(filename):
    data=read_words(filename)
    data_list=''.join(data)
    data_splited=data_list.split('#')
    final_data_list=[]
    for i in data_splited:
        if (not hon(i,lvchu)):
#        if len(i)>20 and (not hon(i,lvchu)):
            final_data_list.append(i)
            
    final_data_string='#'.join(final_data_list)
    final_data=list(final_data_string)
    return final_data
    
#def read_words(filename):
#    with tf.gfile.GFile(filename, "r") as f:
#        data=f.read().decode("utf-8").replace("\n", '#')
#        
#    final_data=[]
#    for i in data:
#        final_data.append(i)
##        if is_chinese(i):
##            final_data.append(i)
##        else:
##            final_data.append(Q2B(i))
#                
#    return final_data


################
def label_build(filename,dict_of_word_ids):
    start=time.time()
           
    """the table of CWS and PU  jointly
    =====================================================================================
          class             ||            label                                        ||        
    =====================================================================================
    padding space           ||              0                                          ||
    =====================================================================================
    the left                ||              1                                          ||
    =====================================================================================
    the middle              ||              2                                          ||
    =====================================================================================
    single character        ||  "," | "." | "?" |  "!" | "、" | ":" | ";" |  "#" |  " "||
    -------------------------------------------------------------------------------------
    label                   ||  3   | 4   | 5   |  6   | 7    | 8   | 9   |  10  |  11 ||
    =====================================================================================
    the right character     ||  "," | "." | "?" |  "!" | "、" | ":" | ";" |  "#" |  " "||
    -------------------------------------------------------------------------------------
    label                   ||  12  | 13  | 14  |  15  | 16   | 17  | 18  |  19  |  20 ||
    =====================================================================================
    
    
    the table of CWS
    ====================================================================================================
          class             ||  padding space ||    left   ||   middle    ||    single   ||   right   ||
    ====================================================================================================
          label             ||         0      ||     1     ||      2      ||       3     ||      4    ||
    ====================================================================================================
    
    
    the table of PU
    ===================================================================================================
    class   || padding space || do nothing || "," || "." || "?" || "!" ||  "、" || ":" || ";" || "#" ||
    ===================================================================================================
    label   ||     0         ||      1     ||  2  ||  3  ||  4  ||  5  ||   6   ||  7  ||  8  ||  9  ||
    ===================================================================================================
    """



#    bound={1:',',2:'。',3:'?',4:'!',5:' '}
#    def predict_label(bound,_temp_data,sor):
#        
#        if sor=='s':
#            _temp_index=0
#        elif sor=='r':
#            _temp_index=len(bound)
#        if (bound[1] in _temp_data):
#            return 2+_temp_index
#        elif (bound[2] in _temp_data):
#            return 3+_temp_index
#        elif (bound[3] in _temp_data):
#            return 4+_temp_index
#        elif (bound[4] in _temp_data):
#            return 5+_temp_index
#        else :#(bound[5] in _temp_data) #and (not hon(temp_data,list(bound.values())[0:-1])): #if the right area do not have elememnt of bound except the last one
#            return 6+_temp_index
        
    bound9=['，',     '。',      '？',      '！',    '、',      '：',      '；',    '#',   ' ']
    special_chinese=['-',   '°',   '±',  '·',     '×',   '—',   '…',   '‰',
                 
                 '℃',  'Ⅱ',  'Ⅲ',  '∶',    '①',  '②',  '③',   '④', 
                 
                 '⑤',  '⑥',  '⑦',  '⑵',   '▲',   '△',   '○',   '●',  
                 
                 '％',  '＋',  '－',  '．',   '／',  '＝',  '＞',   '～']
            
    def predict_label9(bound9,_temp_data,sor):
        if sor=='s':
            _temp_index=0
        elif sor=='r':
            _temp_index=len(bound9)
            
        bound9_dict=dict(zip(range(len(bound9)),bound9))
        for i in range(len(bound9)):
            if bound9_dict[i] in _temp_data:
                return i+3+_temp_index
        

        
        

    data_x=[]
    final_data_y=[]
    data = readword_with_filter_nois_word(filename)
    data=dict(zip(range(len(data)),data))
            
    index_ChineseList=[]
    for i in range(len(data)):
        if (not is_other(data[i])) or (data[i] in special_chinese):
            index_ChineseList.append(i)
#    print('aaaaaaaaaaaaaaaaaaaa',len(index_ChineseList))
            
    index_ChineseList=dict(zip(range(len(index_ChineseList)),index_ChineseList))
    
    def build_tem(data,i,j):
        final_data=[]
        while (j-i)>=0:
            final_data.append(data[i])
            i=i+1
        return final_data
        
    num_chinese=len(index_ChineseList)
    I_previous=0
    for i in range(num_chinese): # in the range of index_ChineseList
        I=index_ChineseList[i]
        data_x.append(data[I])                # add the first chinese character to che data_x list which is a container of chinese character
#        if data[I]==' ':
#            final_data_y.append(2)
#            continue
        
        if i<=len(index_ChineseList)-2:
            I_next= index_ChineseList[i+1] # the next index of i
        
        if i==0:   # the first index ,there is no way that the y_label is the right character or the middle one
        
            temp_data=build_tem(data,I,I_next)
            
            if ((I_next-I)==1):               #if i and i_next are neighbors ,then there have no room for another none-chinese-character
#                final_data_y.append(0)  #the left
                final_data_y.append(1)  #the left
            else:
                final_data_y.append(predict_label9(bound9,temp_data,'s'))
                        
       
        ##################################################################################################
        elif i==num_chinese-1:# the end chinese character(single or right)
            
#            temp_data=data[ i: (len(data)-1) ]
            temp_data=build_tem(data,I,(len(data)-1))
            print('temp_data........',temp_data,'\n')  
            if hon(build_tem(data,I_previous,I),bound9):#single
#            hon(data[i_previous:i],bound)
#                print(temp_data)
                final_data_y.append(predict_label9(bound9,temp_data,'s'))  
                
            else: # right
                
                final_data_y.append(predict_label9(bound9,temp_data,'r'))
                    
        #################################################################################################
        
        #################################################################################################   
        else: # the middle chinese character 
            temp_data=build_tem(data,I,I_next)
#            print(data[i_previous:i])
            if hon(build_tem(data,I_previous,I),bound9): # mabye is the single character or the left one
                
#            hon(data[i:i_next],bound)
                if hon(build_tem(data,I,I_next),bound9): # is the single character
                    
                    final_data_y.append(predict_label9(bound9,temp_data,'s'))
                    
                else: # left
#                    final_data_y.append(0)
                    final_data_y.append(1)  #the left
                    
            else:#maybe is the right character or the middle one
                if hon(build_tem(data,I,I_next),bound9): # is the right character
                    final_data_y.append(predict_label9(bound9,temp_data,'r'))
                        
                else: # the middle character
#                    final_data_y.append(1)  
                    final_data_y.append(2)  
       ################################################################################################ 
        I_previous=I
    
    
#    dict_of_word_ids=convert2dict(data_x)
#    dict_of_word_ids=build_vocab(filename)
#    temp_x=dict(zip(data_x,range(len(data_x))))
    final_data_x=convert_words_to_ids(data_x,dict_of_word_ids)
    final_data=(final_data_x,final_data_y)
    end=time.time()
    print(end-start)
    return final_data

#########################################################################################################                    
            
            
    
def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "pku_training.utf8")
#  valid_path = os.path.join(data_path, "ptb.valid.txt")
#  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = build_vocab(train_path)
  train_data = label_build(train_path, word_to_id)
  
  
#  valid_data = label_build(valid_path, word_to_id)
#  test_data = label_build(test_path, word_to_id)
#  vocabulary = len(word_to_id)
  valid_data=None
  test_data=None
  
  
  return train_data,valid_data,test_data

def ptb_raw_data_with_saved_dict(data_path_train,data_path_test=None,dict_data=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path_train, "pku_training.utf8")
#  train_path = os.path.join(data_path_train, "control.utf8")
  word_to_id=dict_data or build_vocab(train_path)
  
  train_data = label_build(train_path, word_to_id)
#  valid_data = label_build(valid_path, word_to_id)
#  test_data = label_build(test_path, word_to_id)
#  vocabulary = len(word_to_id)
  valid_data=None
  
  if data_path_test is not None:
      
      test_path = os.path.join(data_path_test, "pku_test_gold.utf8")
#      test_path = os.path.join(data_path_test, "control.utf8")
      test_data = label_build(test_path, word_to_id)
  else:
      test_data=None
  
  
  return train_data,valid_data,test_data

bound9=['，',     '。',      '？',      '！',    '、',      '：',      '；',    '#',   ' ']

def split_data_with_sentense_separator(data_x,data_y,num_step,
#                          index_bound={'#':50,'；':60,'。':70,'？':80,'！':90,'，':90,' ':90},
                            index_bound=None,
                          bound9=['，',  '。',  '？',  '！',  '、',  '：', '；',  '#',  ' ']):
    space_c=dict_data[' ']
    sentence_separator=list(index_bound.keys())
    bound9_index_dict=dict(zip(bound9,range(len(bound9))))
    
    
    def find_index(y,num):#to find out whether the list y have these separators which is the bound9's first num element
        num_bound=len(bound9)
        accumulate_bound=[]
        for c in sentence_separator[0:num]:
            bound_label=[3+bound9_index_dict[c],3+num_bound+bound9_index_dict[c]]
            accumulate_bound.append(bound_label[0])
            accumulate_bound.append(bound_label[1])
            
#        accumulate_bound=set(accumulate_bound)
        for i in accumulate_bound:
            if i in y:
                return y.index(i)
        return None    

    final_data_x=[]
    final_data_y=[]
    flow_begain=0
    flow_end=num_step
    num_data=len(data_y)
    index_bound=list(index_bound.values())
#    index_bound=index_bound[::-1]
    while flow_end<=num_data:
    
        y=data_y[flow_begain : flow_end]

        y=y[::-1]
        temp_count=0
        temp_index=None
        for len_form_end in index_bound:
            temp_y=y[0:len_form_end]
#            print(temp_y)
            temp_count=temp_count+1
            temp_index=find_index(temp_y,temp_count)
            if temp_index!=None:
                break
#            print(temp_index)
        the_end=flow_end - temp_index       
        temp_data_y=data_y[flow_begain : the_end]
#        temp_data_y.extend([2]*(num_step-len(temp_data_y)))  # padding y ,2 is the label of the representation of space
        temp_data_y.extend([0]*(num_step-len(temp_data_y)))  # padding y ,0 is the label of the representation of space
        final_data_y.append(temp_data_y)
        
        temp_data_x=data_x[flow_begain : the_end]
        temp_data_x.extend([space_c]*(num_step-len(temp_data_x)))  #padding the x with label of space
        final_data_x.append(temp_data_x)
        
        flow_begain=flow_end - temp_index
        flow_end=flow_begain+num_step
        if flow_end>num_data:
            flow_end=num_data
            break
        
    y=data_y[flow_begain : flow_end]
    
    temp_data_y=data_y[flow_begain : flow_end]    
#    temp_data_y.extend([2]*(num_step-len(temp_data_y)))
    temp_data_y.extend([0]*(num_step-len(temp_data_y))) # padding y ,0 is the label of the representation of space
    final_data_y.append(temp_data_y)
    
    temp_data_x=data_x[flow_begain : flow_end]
    temp_data_x.extend([space_c]*(num_step-len(temp_data_x)))
    final_data_x.append(temp_data_x)
    return final_data_x,final_data_y


bound9=['，',     '。',      '？',      '！',    '、',      '：',      '；',    '#',   ' ']


def ptb_producer_IO(raw_data, batch_size, num_steps, interval_size=100,name=None,
#                    index_bound={'#':500,'；':500,'。':500,'？':500,'！':500,'，':500}
                     index_bound=None
                                                                                       ):
    
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    
    (raw_data_x,raw_data_y)=raw_data
#    print(raw_data_y)
    raw_data_x,raw_data_y=split_data_with_sentense_separator(raw_data_x,raw_data_y,num_steps,
                                                             index_bound=index_bound)
    raw_data_x=sum(raw_data_x,[])
    raw_data_y=sum(raw_data_y,[])
    
    
    epoch_size=(len(raw_data_x)-num_steps*batch_size)//(interval_size*batch_size)+1+1
    return_epoch_size=epoch_size
    batch_len=interval_size*(epoch_size-1)+num_steps
    num_padding=interval_size*batch_size*(epoch_size-1)+num_steps*batch_size-len(raw_data_x)
    
    raw_data_y.extend([0]*num_padding)
    raw_data_x.extend([dict_data[' ']]*num_padding)
    
    
    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

#    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
    
    raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.int32)
    data_x = tf.reshape(raw_data_x[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    
    raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.int32)
    data_y = tf.reshape(raw_data_y[0 : batch_size * batch_len],
                  [batch_size, batch_len])
#    x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
    
    x = tf.strided_slice(data_x, [0, i * interval_size ], [batch_size, i * interval_size + num_steps])
    y = tf.strided_slice(data_y, [0, i * interval_size ], [batch_size, i * interval_size + num_steps])
   
    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])
    
    
  return x,y,return_epoch_size


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    
    (raw_data_x,raw_data_y)=raw_data
    if raw_data_y!=None:
        
        raw_data_x,raw_data_y=split_data_with_sentense_separator(raw_data_x,raw_data_y)
        raw_data_x=sum(raw_data_x,[])
        raw_data_y=sum(raw_data_y,[])
        
        
        epoch_size_before=(len(raw_data_x) // batch_size)//num_steps
        num_remaining=len(raw_data_x)-(epoch_size_before*num_steps*batch_size)
        num_padding=batch_size*num_steps-num_remaining
        
        raw_data_y.extend([0]*num_padding)
    else:
        raw_data_x=sum(raw_data_x,[])
        epoch_size_before=(len(raw_data_x) // batch_size)//num_steps
        num_remaining=len(raw_data_x)-(epoch_size_before*num_steps*batch_size)
        num_padding=batch_size*num_steps-num_remaining
        
    raw_data_x.extend([dict_data[' ']]*num_padding)
#    data_len = tf.size(raw_data_x)
#    batch_len = data_len // batch_size
#    epoch_size = (batch_len - 1) // num_steps
    print('len(raw_data_x)..........sssssssssssssssssssssss',len(raw_data_x))
    
    batch_len=len(raw_data_x) // batch_size
    epoch_size=batch_len//num_steps
    
    
    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
#    i = tf.train.range_input_producer(batch_len-num_steps, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
    
    
    raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.int32)
    data_x = tf.reshape(raw_data_x[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#    x = tf.strided_slice(data_x, [0, i ], [batch_size, i+num_steps])
    x.set_shape([batch_size, num_steps])
    
    y=None
    if raw_data_y!=None:
        raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.int32)
        data_y = tf.reshape(raw_data_y[0 : batch_size * batch_len],
                      [batch_size, batch_len])
        y = tf.strided_slice(data_y, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#        y = tf.strided_slice(data_y, [0, i], [batch_size, i+num_steps])
        y.set_shape([batch_size, num_steps])
    
    return x,y,batch_len//num_steps




#def ptb_producer(raw_data, batch_size, num_steps, name=None):
#    
#  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
#    
#    (raw_data_x,raw_data_y)=raw_data
#    if raw_data_y!=None:
#        raw_data_x,raw_data_y=split_data_with_sentense_separator(raw_data_x,raw_data_y)
#        
#        num_remaining=len(raw_data_x)%(batch_size)
#        num_padding=(batch_size-num_remaining)*num_steps
#        
#        raw_data_y=sum(raw_data_y,[])
#        raw_data_y.extend([0]*num_padding)
#        
#    raw_data_x=sum(raw_data_x,[])
#    raw_data_x.extend([dict_data[' ']]*num_padding)
##    data_len = tf.size(raw_data_x)
##    batch_len = data_len // batch_size
##    epoch_size = (batch_len - 1) // num_steps
#    print('len(raw_data_x)..........sssssssssssssssssssssss',len(raw_data_x))
#    
#    batch_len=len(raw_data_x) // batch_size
#    epoch_size=batch_len//num_steps
#    
#    
#    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
#    with tf.control_dependencies([assertion]):
#      epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
##    i = tf.train.range_input_producer(batch_len-num_steps, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
#    
#    
#    raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.int32)
#    data_x = tf.reshape(raw_data_x[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#    x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
##    x = tf.strided_slice(data_x, [0, i ], [batch_size, i+num_steps])
#    x.set_shape([batch_size, num_steps])
#    
#    y=None
#    if raw_data_y!=None:
#        raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.int32)
#        data_y = tf.reshape(raw_data_y[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#        y = tf.strided_slice(data_y, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
##        y = tf.strided_slice(data_y, [0, i], [batch_size, i+num_steps])
#        y.set_shape([batch_size, num_steps])
#    
#    return x,y,batch_len//num_steps


#def ptb_producer(raw_data, batch_size, num_steps, name=None):
#  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
#    
#    (raw_data_x,raw_data_y)=raw_data
#    data_len = tf.size(raw_data_x)
#    batch_len = data_len // batch_size
#    epoch_size = (batch_len - 1) // num_steps
#    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
#    with tf.control_dependencies([assertion]):
#      epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
#    
#    raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.int32)
#    data_x = tf.reshape(raw_data_x[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#    x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#    x.set_shape([batch_size, num_steps])
#    
#    y=None
#    if raw_data_y!=None:
#        raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.int32)
#        data_y = tf.reshape(raw_data_y[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#        y = tf.strided_slice(data_y, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#        y.set_shape([batch_size, num_steps])
#    
#    return x,y


#def ptb_producer(raw_data, batch_size, num_steps, name=None):
#  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
#      
#    (raw_data_x,raw_data_y)=raw_data
#    raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.int32)
#    raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.int32)
#
#    data_len = tf.size(raw_data_x)
#    batch_len = data_len // batch_size
#    
#    data_x = tf.reshape(raw_data_x[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#    
#    data_y = tf.reshape(raw_data_y[0 : batch_size * batch_len],
#                      [batch_size, batch_len])
#    
#    
#
#    epoch_size = (batch_len - 1) // num_steps
#    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
#    with tf.control_dependencies([assertion]):
#      epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()#produce 0,1,2......epoch_size-1
#    
#    x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#    x.set_shape([batch_size, num_steps])
#    
#    y = tf.strided_slice(data_y, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#    y.set_shape([batch_size, num_steps])
#    return x , y
    
    
    
    

def get_only(str):
    new_str=[]
    for i in str:
        if not (i in new_str):
            new_str.append(i)
    return new_str

def get_diffenrent(str1,str2):
    r1=[]
    r2=[]
    for i in str1:
        if not (i in str2):
            r1.append(i)
            
    for i in str2:
        if not (i in str1):
            r2.append(i)
    return r1,r2


def hon_diff_QB(str):
    new_str=[]
    for i in str:
        if B2Q(i)==i:
            new_str.append(i)
    return new_str




#flags.DEFINE_string("data_path", '/home/li/Desktop/TF/my/icwb2_data/training/',
#                    "Where the training/test data is stored.")
#
#if __name__=="__main__":
#    
#    train_data,valid_data,test_data=ptb_raw_data(FLAGS.data_path)
#    x,y=train_data
##a,b=ptb_producer(train_data,30,100)


