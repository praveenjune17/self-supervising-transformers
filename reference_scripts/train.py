# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_one_hot_labels, monitor_run
from utilities import log
from create_model import source_tokenizer, target_tokenizer
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set, training_results)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.train_batch_size
                              )
val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             drop_remainder=True                          
                             )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)
for (step, (input_ids, target_ids_)) in tqdm(enumerate(train_dataset), initial=1):
    start=time.time()
    draft_mask, refine_mask, target_ids = mask_and_one_hot_labels(target_ids_)
    grad_accum_flag = True if ((step+1)%config.gradient_accumulation_steps) == 0 else False
    refine_predictions = train_step(
                                    input_ids,  
                                    target_ids_, 
                                    target_ids, 
                                    draft_mask,
                                    refine_mask,
                                    grad_accum_flag
                                    )
    if grad_accum_flag:
        train_loss = batch_run_check(
                                  step+1,  
                                  start
                                  )
    evaluate = ((step+1) * config.train_batch_size) % config.eval_after
    if evaluate == 0:
        predicted = train_sanity_check(target_tokenizer, refine_predictions, target_ids_)
        ckpt_save_path = ck_pt_mgr.save()
        if predicted:
            (rouge_score, bert_score) = evaluate_validation_set(
                                                              val_dataset, 
                                                              step+1
                                                              )
        else:
            rouge_score, bert_score = 0
        training_results(
                              step+1, 
                              rouge_score, 
                              bert_score,
                              (time.time() - start),
                              ckpt_save_path
                              )
        monitor_early_stop = monitor_run(
                                        ckpt_save_path, 
                                        bert_score, 
                                        rouge_score,
                                        train_loss, 
                                        step+1
                                        )
        if not monitor_early_stop:
            break
log.info(f'Training completed at step {step+1}')