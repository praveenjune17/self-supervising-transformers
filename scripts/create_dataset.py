import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
from functools import partial
import os
import six
from utilities import log
from preprocess import preprocess_dataset
from configuration import config, source_tokenizer, target_tokenizer

'''
borrowed from https://github.com/tensorflow/models/official/nlp/transformer/data_pipeline.py
'''
def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in six.iteritems(dictionary):
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))

def download_and_encode(record_file_path, source_tokenizer, target_tokenizer, split='test'):
    
    if not os.path.exists(record_file_path):
        if config.tfds_name == 'en_tam_parallel_text':
            en_tam_ds = defaultdict(list)
            record_count=0
            #List of available datasets in the package
            dataset_names = ['GNOME_v1_en_to_ta', 'GNOME_v1_en_AU_to_ta', 'GNOME_v1_en_CA_to_ta', 
                     'GNOME_v1_en_GB_to_ta', 'GNOME_v1_en_US_to_ta', 'KDE4_v2_en_to_ta', 
                     'KDE4_v2_en_GB_to_ta', 'Tatoeba_v20190709_en_to_ta', 'Ubuntu_v14.10_en_to_ta_LK', 
                     'Ubuntu_v14.10_en_GB_to_ta_LK', 'Ubuntu_v14.10_en_AU_to_ta_LK', 'Ubuntu_v14.10_en_CA_to_ta_LK', 
                     'Ubuntu_v14.10_en_US_to_ta_LK', 'Ubuntu_v14.10_en_to_ta', 'Ubuntu_v14.10_en_GB_to_ta', 
                     'Ubuntu_v14.10_en_AU_to_ta', 'Ubuntu_v14.10_en_CA_to_ta', 'Ubuntu_v14.10_en_NZ_to_ta', 
                     'Ubuntu_v14.10_en_US_to_ta', 'OpenSubtitles_v2018_en_to_ta', 'OpenSubtitles_v2016_en_to_ta',
                     'en_ta', 'github_joshua_en_ta']
            for name in dataset_names:
                en_tam_ds[(name,'metadata_'+name)] = tfds.load('en_tam_parallel_text'+'/'+name, 
                                                              with_info=True, 
                                                              as_supervised=True,
                                                              data_dir=config.tfds_data_dir,
                                                              builder_kwargs={'version': None},
                                                            )
            for i,j  in en_tam_ds.keys():
                record_count+=(sum([i.num_examples for i in  list(en_tam_ds[(i, j)][1].splits.values())]))
            if not config.test_script:
                log.info(f'Total record count without filtering is {record_count}')
            #initialize the first dataset to the train_examples variable
            #Concatenate all the train datasets
            if split == 'train':
                downloaded_tfd = en_tam_ds[('GNOME_v1_en_to_ta', 'metadata_GNOME_v1_en_to_ta')][0]['train']
                for typ in list(en_tam_ds.keys())[1:]:
                    downloaded_tfd = downloaded_tfd.concatenate(en_tam_ds[typ][0]['train'])
              #other splits include validation and test 
            else:
                downloaded_tfd = en_tam_ds[('en_ta', 'metadata_en_ta')][0][split]
        else:
            downloaded_tfd, ds_info = tfds.load(
                                 config.tfds_name, 
                                 with_info=True,
                                 as_supervised=True, 
                                 data_dir=config.tfds_data_dir,
                                 builder_kwargs={'version': config.tfds_data_version},
                                 split=tfds.core.ReadInstruction(split, from_=from_, to=to, unit='%')
                                )
            record_count = (sum([i.num_examples for i in  list(ds_info.splits.values())]))
        with tf.io.TFRecordWriter(record_file_path) as writer:
            for (input_line, target_line) in (downloaded_tfd):
                input_line, target_line = input_line.numpy().decode('utf-8'), target_line.numpy().decode('utf-8')
                example = dict_to_example(
                    {"inputs": source_tokenizer.encode(input_line,
                                            add_special_tokens=True),
                     "targets": target_tokenizer.encode(target_line,
                                                add_special_tokens=True)
                    }
                )
                writer.write(example.SerializeToString())
    else:
        print('Dataset already dowloaded and encoded')


datasets = {}
for split in ['train', 'validation']:
    if split == 'train':
        drop_remainder=False
        shuffle=True
        batch_size = config.train_batch_size
        num_examples_to_select = config.samples_to_train
    else:
        drop_remainder=True
        shuffle=False
        batch_size = config.validation_batch_size
        num_examples_to_select = config.samples_to_validate
    records_file_path = os.path.join(config.tf_records_path, f'{config.tfds_name}_{split}.tfrecords')
    download_and_encode(records_file_path, source_tokenizer, target_tokenizer, split)
    datasets[split] = preprocess_dataset(record_file_path=records_file_path,
                                         drop_remainder=drop_remainder,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_examples_to_select=num_examples_to_select)
    print(split, 'dataset created')

train_dataset = datasets['train']
val_dataset = datasets['validation']