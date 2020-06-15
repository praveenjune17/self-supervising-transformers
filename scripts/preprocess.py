# -*- coding: utf-8 -*-
import tensorflow as tf
from configuration import config

parallel_calls = config.num_parallel_calls
def tfpad(ids, ids_length, pad=0):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_len = tf.maximum(0, ids_length - tf.shape(ids)[-1])
    pad_with = tf.reshape(tf.concat([[0], [1]], axis=0)*pad_len, (1,2))

    return tf.pad(ids, pad_with, "CONSTANT", constant_values=pad)

def tf_padtf_encoded_ids(input_ids, target_ids, 
          input_seq_len=config.input_seq_length, 
          output_seq_len=config.target_seq_length):
    # Account for [CLS] and [SEP] with "- 2"
    input_ids = tf.cond(tf.math.greater(
                                          tf.shape(input_ids)[-1],
                                          input_seq_len-2
                                          ), lambda: input_ids[0:(input_seq_len - 2)], 
                                             lambda: input_ids
                            )
    target_ids = tf.cond(tf.math.greater(
                                          tf.shape(target_ids)[-1],
                                          output_seq_len+1-2
                                          ), lambda: target_ids[0:(output_seq_len+1 - 2)], 
                                             lambda: target_ids
                            )
    input_ids = tfpad(input_ids, input_seq_len)
    target_ids = tfpad(target_ids, output_seq_len+1)

    return input_ids, target_ids

# Set threshold for input_sequence and  output_sequence length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.math.count_nonzero(x) <= config.input_seq_length,
                          tf.math.count_nonzero(y) <= config.target_seq_length
                         )

def _parse_example(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    data_fields = {
      "inputs": tf.io.VarLenFeature(tf.int64),
      "targets": tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])
    return inputs, targets

def preprocess_dataset(
                   record_file_path,
                   shuffle,
                   batch_size,
                   num_examples_to_select,
                   drop_remainder,
                   parallel_calls=-1):

    tf_dataset = tf.data.TFRecordDataset(record_file_path)
    tf_dataset = tf_dataset.map(_parse_example, num_parallel_calls=-1)
    tf_dataset = tf_dataset.filter(filter_max_length)
    tf_dataset = tf_dataset.map(tf_padtf_encoded_ids)
    #tf_dataset = tf_dataset.repeat(20000) 
    if shuffle:
        tf_dataset = tf_dataset.shuffle(1000000, seed=101)
    tf_dataset = tf_dataset.take(num_examples_to_select) 
    #tf_dataset = tf_dataset.take(49214+45000+7200+12240+7200+7200)
    tf_dataset = tf_dataset.cache()
    tf_dataset = tf_dataset.padded_batch(batch_size, 
                            padded_shapes=([-1], [-1]), 
                            drop_remainder=drop_remainder)
    tf_dataset = tf_dataset.prefetch(buffer_size=parallel_calls)
    
    return tf_dataset
