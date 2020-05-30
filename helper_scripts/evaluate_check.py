# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, 'D:\\Local_run\\Distributed_Training_Self-supervising-transformers\\scripts')
sys.path.insert(0, 'D:\\Local_run\\models')
import tensorflow as tf
tf.random.set_seed(100)
import time
from tqdm import tqdm
from create_dataset import val_dataset
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log
from model_training_helper import (check_ckpt, evaluate_validation_set, training_results)


step = 1
ck_pt_mgr = check_ckpt(config.checkpoint_path)
start_time = time.time()
(task_score, bert_score,
  draft_attention_weights,
  refine_attention_weights) = evaluate_validation_set(       
                                                    val_dataset.take(1),
                                                    step
                                                    )  
training_results(
                  step,
                  0,
                  0,
                  task_score, 
                  bert_score,
                  (time.time() - start_time),
                  'none',
                  log,
                  config
                  )
