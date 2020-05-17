# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import shutil
import os
from configuration import config
from rouge import Rouge
from create_model import target_tokenizer 
from bert_score import score as b_score
from utilities import log, valid_output_sequence_writer, detokenize


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
      super(CustomSchedule, self).__init__()
      
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps
      
    def __call__(self, step):
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def label_smoothing(inputs, epsilon=config.epsilon_ls):
    # V <- number of channels
    V = inputs.get_shape().as_list()[-1] 
    epsilon = tf.cast(epsilon, dtype=inputs.dtype)
    V = tf.cast(V, dtype=inputs.dtype)
    return ((1-epsilon) * inputs) + (epsilon / V)

def mask_and_one_hot_labels(target):
    draft_mask = tf.math.logical_not(tf.math.equal(target[:, 1:], config.PAD_ID))
    refine_mask = tf.math.logical_not(tf.math.logical_or(tf.math.equal(target[:, :-1], config.target_CLS_ID), 
                                                         tf.math.equal(target[:, :-1], config.PAD_ID)
                                                         )
                                      )
    target_ids_3D = label_smoothing(tf.one_hot(target, depth=config.target_vocab_size))
    return (draft_mask, refine_mask, target_ids_3D)

def convert_wordpiece_to_words(w_piece):
    new=[]
    for i in w_piece:
        if '##' in i:
            m = i.replace('##', '')
        else:
            if w_piece.index(i) == 0:
                m = i
            else:
                m = ' '+i
        new.append(m)
    return (''.join(new))

def loss_function(real, pred, mask):
    # pred shape == real shape = (batch_size, tar_seq_len, target_vocab_size)
    loss_object = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
    loss_  = loss_object(real, pred)
    mask   = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def get_loss_and_accuracy():
    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')
    return(loss, accuracy)
    
def write_output_sequence(tar_real, predictions, step, write_output_seq):
    ref_sents = []
    hyp_sents = []
    rouge_all = Rouge()
    for tar, ref_hyp in zip(tar_real, predictions):
        detokenized_refs, detokenized_hyp_sents = detokenize(target_tokenizer, 
                                                           tf.squeeze(tar), 
                                                           tf.squeeze(ref_hyp) 
                                                           )
        ref_sents.append(detokenized_refs)
        hyp_sents.append(detokenized_hyp_sents)
    try:
        rouges = rouge_all.get_scores(ref_sents , hyp_sents)
        avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                                        rouge_scores['rouge-2']["f"], 
                                        rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
        _, _, bert_f1 = b_score(ref_sents, hyp_sents, model_type=config.bert_score_model)
        avg_bert_f1 = np.mean(bert_f1.numpy())
    except:
        log.warning('Some problem while calculating ROUGE so setting ROUGE score to zero')
        avg_rouge_f1 = 0
        avg_bert_f1 = 0
    
    if write_output_seq:
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy()), 'w') as f:
            for ref, hyp in zip(ref_sents, hyp_sents):
                f.write(ref+'\t'+hyp+'\n')
    return (avg_rouge_f1, avg_bert_f1)
  
  
def tf_write_output_sequence(tar_real, predictions, step, write_output_seq):
    return tf.py_function(write_output_sequence, 
                          [tar_real, predictions, step, write_output_seq], 
                          Tout=[tf.float32, tf.float32]
                          )
    

def monitor_run(ckpt_save_path, 
                bert_score, 
                rouge_score, 
                train_loss,
                step,
                to_monitor=config.monitor_metric):
  
    ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
    if config.run_tensorboard:
        with valid_output_sequence_writer.as_default():
            tf.summary.scalar('ROUGE_f1', rouge_score, step=step)
            tf.summary.scalar('BERT_f1', bert_score, step=step)
    monitor_metrics = dict()
    monitor_metrics['BERT_f1'] = bert_score
    monitor_metrics['ROUGE_f1'] = rouge_score
    monitor_metrics['combined_metric'] = (
                                          monitor_metrics['BERT_f1'], 
                                          monitor_metrics['ROUGE_f1']
                                          )
    # multiply with the weights                                    
    monitor_metrics['combined_metric'] = round(
                                        tf.reduce_sum([(i*j) for i,j in zip(monitor_metrics['combined_metric'],  
                                                                            config.combined_metric_weights)]).numpy(), 
                                                                            2
                                                                            )
    log.info(f"combined_metric {monitor_metrics['combined_metric']:4f}")
    if config.last_recorded_value < monitor_metrics[to_monitor]:
        # reset tolerance to zero if the monitor_metric decreases before the tolerance threshold
        config.tolerance=0
        config.last_recorded_value =  monitor_metrics[to_monitor]
        ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                             if ckpt_string in files]
        log.info(f'{to_monitor} is {monitor_metrics[to_monitor]:4f} so checkpoint files {ckpt_string} \
                 will be copied to best checkpoint directory')
        # copy the best checkpoints
        shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), config.best_ckpt_path)
        for files in ckpt_files_tocopy:
            shutil.copy2(os.path.join(ckpt_fold, files), config.best_ckpt_path)
    else:
        config.tolerance+=1
    # stop if minimum training loss is reached
    if train_loss < config.min_train_loss:
        log.info(f'Stop training since minimum training loss reached')
        return False
    else:
        return True
    # Warn and early stop
    if config.tolerance > config.tolerance_threshold:
        log.warning('Tolerance exceeded')
        if config.early_stop:
            log.info(f'Early stopping since the {to_monitor} reached the tolerance threshold')
            return False
        else:
            return True
    else:
        return True

def get_optimizer():
    lr = config.learning_rate if config.learning_rate else CustomSchedule(config.d_model)    
    if config.grad_clipnorm:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=lr, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 clipnorm=config.grad_clipnorm,
                                 epsilon=1e-9
                                 )
    else:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=lr, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 epsilon=1e-9
                                 )
    return optimizer

optimizer = get_optimizer()
_ = b_score(["I'm Batman"], ["I'm Spiderman"], lang='en', model_type=config.target_pretrained_bert_model)
log.info('Loaded Pre-trained BERT for BERT SCORE calculation')