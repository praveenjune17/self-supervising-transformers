# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, 'D:\\Local_run\\Distributed_Training_Self-supervising-transformers\\scripts')
sys.path.insert(0, 'D:\\Local_run\\models')
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
#tf.keras.backend.clear_session()
tf.random.set_seed(100)

import time
from tqdm import tqdm
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check,
                          train_sanity_check)
from create_dataset import train_dataset, val_dataset

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)
stop_at = 1000

for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):
    start_time = time.time()
    grad_accum_flag = (True if (step%config.gradient_accumulation_steps) == 0 else False) if config.accumulate_gradients else None
    predictions, bert_f1_score = train_step(
                            input_ids,  
                            target_ids,
                            grad_accum_flag
                            )
    if (step % config.steps_to_print_training_info) == 0:
        train_loss = batch_run_check(
                                  step,  
                                  start_time,
                                  bert_f1_score
                                  )
        train_sanity_check(target_tokenizer, predictions, target_ids, log)
    if (step % stop_at) == 0:
        break
train_sanity_check(target_tokenizer, predictions, target_ids, log)
log.info(f'Training completed at step {step}')
