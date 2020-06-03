import tensorflow as tf
import os
import six
from utilities import log
from configuration import config

def preprocess(split, batch_size):
    
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
    # Set threshold for input_sequence and  output_sequence length
    def filter_max_length(x, y):
        return tf.logical_and(
                              tf.math.count_nonzero(x) <= config.input_seq_length,
                              tf.math.count_nonzero(y) <= config.target_seq_length
                             )
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

    record_file_path = os.path.join(config.tf_records_path, f'{config.tfds_name}_{split}.tfrecords')
    tf_dataset = tf.data.TFRecordDataset(record_file_path)
    tf_dataset = tf_dataset.map(_parse_example, num_parallel_calls=-1)
    tf_dataset = tf_dataset.filter(filter_max_length)
    tf_dataset = tf_dataset.map(tf_padtf_encoded_ids)
    tf_dataset = tf_dataset.cache()
    tf_dataset = tf_dataset.padded_batch(batch_size, 
                            padded_shapes=([-1], [-1]), 
                            drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(buffer_size=-1)
    
    return tf_dataset
