# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from create_dataset import train_dataset, val_dataset
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check, 
                          save_evaluate_monitor)

skip = 115253
# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.skip(skip)

try:
    for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=skip+1):
        start_time = time.time()
        grad_accum_flag = (True if (step%config.gradient_accumulation_steps) == 0 else False) if config.accumulate_gradients else None
        predictions, bert_f1_score = train_step(
                                input_ids,  
                                target_ids,
                                grad_accum_flag
                                )
        if (step % config.steps_to_print_training_info) == 0 and grad_accum_flag:
            batch_run_check(
                            step,  
                            start_time,
                            bert_f1_score
                            )
        if (step % config.eval_after_steps) == 0 and grad_accum_flag:
            (early_stop,
            draft_attention_weights,
            refine_attention_weights) = save_evaluate_monitor(ck_pt_mgr, val_dataset, 
                                                    target_tokenizer, predictions, 
                                                    target_ids, step, start_time,
                                                    bert_f1_score
                                                    )
            if early_stop:
                break
        else:
            early_stop = True
    if not early_stop:
        _ = save_evaluate_monitor(ck_pt_mgr, val_dataset, 
                target_tokenizer, predictions, target_ids, step, start_time, bert_f1_score)
    log.info(f'Training completed at step {step}')
except KeyboardInterrupt:
    log.info(f' Checkpoint saved due to KeyboardInterrupt at step {step} in {ck_pt_mgr.save()}')
