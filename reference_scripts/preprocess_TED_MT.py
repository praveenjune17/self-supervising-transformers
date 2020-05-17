# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from functools import partial
from collections import defaultdict
from configuration import config
from utilities import log

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pad(l, n, pad=config.PAD_ID):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_with = (0, max(0, n - len(l)))
    return np.pad(l, pad_with, mode='constant', constant_values=pad)


def encode(sent_1, sent_2, source_tokenizer, target_tokenizer, input_seq_len, output_seq_len):
    
    if config.model_architecture == 'bertified_transformer':
        input_ids = [config.input_CLS_ID] + source_tokenizer.encode(sent_1.numpy().decode('utf-8'), 
                                            add_special_tokens=False) + [config.input_SEP_ID]
        target_ids = [config.target_CLS_ID] + target_tokenizer.encode(sent_2.numpy().decode('utf-8'), 
                                            add_special_tokens=False) + [config.target_SEP_ID]
        # Account for [CLS] and [SEP] with "- 2"
        if len(input_ids) > input_seq_len - 2:
            input_ids = input_ids[0:(input_seq_len - 2)]
        if len(target_ids) > (output_seq_len + 1) - 2:
            target_ids = target_ids[0:((output_seq_len + 1) - 2)]
        input_ids = pad(input_ids, input_seq_len)
        target_ids = pad(target_ids, output_seq_len + 1)
    else:    
      input_ids = [config.input_CLS_ID] + source_tokenizer.encode(sent_1.numpy()) + [config.input_SEP_ID]
      target_ids = [config.target_CLS_ID] + target_tokenizer.encode(sent_2.numpy()) + [config.target_SEP_ID]
    return input_ids, target_ids


def tf_encode(source_tokenizer, target_tokenizer, input_seq_len, output_seq_len):
    """
    Operations inside `.map()` run in graph mode and receive a graph
    tensor that do not have a `numpy` attribute.
    The tokenizer expects a string or Unicode symbol to encode it into integers.
    Hence, you need to run the encoding inside a `tf.py_function`,
    which receives an eager tensor having a numpy attribute that contains the string value.
    """    
    def f(s1, s2):
        encode_ = partial(
                          encode, 
                          source_tokenizer=source_tokenizer, 
                          target_tokenizer=target_tokenizer, 
                          input_seq_len=input_seq_len, 
                          output_seq_len=output_seq_len
                          )
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32])
    return f

# Set threshold for input_sequence and  output_sequence length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.math.count_nonzero(x) <= config.input_seq_length,
                          tf.math.count_nonzero(y) <= config.target_seq_length
                         )

def filter_combined_length(x, y):
    return tf.math.less_equal(
                              (tf.math.count_nonzero(x) + tf.math.count_nonzero(y)), 
                              config.max_tokens_per_line
                             )
          
    
def read_csv(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['input_sequence', 'output_sequence']]
    assert len(df.columns) == 2, 'column names should be input_sequence and output_sequence'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["input_sequence"].values, df["output_sequence"].values)

def create_dataset(split, 
                   source_tokenizer, 
                   target_tokenizer, 
                   from_, 
                   to, 
                   batch_size,
                   buffer_size=None,
                   use_tfds=True, 
                   csv_path=None,
                   drop_remainder=False,
                   num_examples_to_select=config.samples_to_test):

  
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    
    #initialize the first dataset to the train_examples variable
    #Concatenate all the train datasets
    if split == 'train':
        raw_dataset = train_examples
    elif split == 'validation':
        raw_dataset = val_examples
          
    tf_dataset = raw_dataset.map(
                                 tf_encode(
                                          source_tokenizer,
                                          target_tokenizer, 
                                          config.input_seq_length, 
                                          config.target_seq_length
                                          ), 
                                 num_parallel_calls=AUTOTUNE
                                 )
    tf_dataset = tf_dataset.filter(filter_max_length)
    tf_dataset = tf_dataset.take(num_examples_to_select) 
    tf_dataset = tf_dataset.cache()
    if buffer_size:
        tf_dataset = tf_dataset.shuffle(buffer_size, seed = 100)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]), drop_remainder=drop_remainder)
    tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
    log.info(f'{split} tf_dataset created')
    return tf_dataset
