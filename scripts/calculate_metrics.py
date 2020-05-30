# -*- coding: utf-8 -*-
import tempfile
import tensorflow as tf
import numpy as np
from rouge import Rouge
from bert_score import score as b_score
from create_model import Model
from model_utils import create_pretrained_model_mask
from official.nlp.transformer import compute_bleu
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log

negative_log_liklihood = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )

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

def label_smoothing(inputs, epsilon):
    # V <- number of channels
    V = inputs.get_shape().as_list()[-1] 
    epsilon = tf.cast(epsilon, dtype=inputs.dtype)
    V = tf.cast(V, dtype=inputs.dtype)

    return ((1-epsilon) * inputs) + (epsilon / V)

def mask_and_calculate_nll_loss(predictions, 
                            target_ids, 
                            mask_a_with,
                            mask_b_with=config.PAD_ID,
                            epsilon=config.epsilon_ls
                            ):

    target_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size), epsilon)
    loss = negative_log_liklihood(target_ids_3D, predictions)
    mask = create_pretrained_model_mask(target_ids, mask_a_with, mask_b_with) 
    loss = loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss, target_ids_3D


def calculate_bert_f1(target_ids, predicted):

    target_mask = create_pretrained_model_mask(target_ids)
    predicted_return_mask = create_pretrained_model_mask(predicted)
    mask = tf.concat([target_mask, predicted_return_mask], axis=0)
    ids = tf.concat([target_ids, predicted], axis=0)
    # (8*batch_size, *_seq_len, target_vocab_size)
    embeddings = Model.decoder_bert_model(ids, attention_mask=mask)[0]
    embeddings_normalized = embeddings/(tf.norm(embeddings, axis=-1)[:, :, tf.newaxis])
    # (4*batch_size,              (4*batch_size, 
    #  tar_seq_len,               cand_seq_len ,
    #  target_vocab_size),        target_vocab_size)
    target_embeddings_normalized, predicted_embeddings_normalized = tf.split(embeddings_normalized, 2, axis=0)
    # (4*batch_size, tar_seq_len, cand_seq_len)
    scores = tf.matmul(predicted_embeddings_normalized, target_embeddings_normalized, transpose_b=True)
    mask = tf.matmul(predicted_return_mask[:, :, tf.newaxis], target_mask[:, tf.newaxis,: ])
    scores = scores*mask
    recall = tf.reduce_max(scores, 1)
    precision = tf.reduce_max(scores, 2)
    recall = tf.reduce_sum(recall, 1)
    precision = tf.reduce_sum(precision, 1)
    recall = recall/tf.reduce_sum(target_mask, -1)
    precision = precision/tf.reduce_sum(predicted_return_mask, -1)
    f1_score = (2*(precision*recall))/(precision+recall)
    
    return f1_score

def calculate_policy_gradient_loss(
                          draft_logits,
                          refine_logits, 
                          sample_returns, 
                          target_ids,
                          candidate_returns,
                          nll_loss,
                          gamma=config.gamma
                          ):
    
    actual_batch_size = tf.shape(target_ids)[0]/4
    (draft_sample_returns, refine_sample_returns) = tf.split(sample_returns, num_or_size_splits=2, axis=0)
    draft_sample_return_nll_loss, _ = mask_and_calculate_nll_loss(draft_logits,
                                                       draft_sample_returns,
                                                       config.CLS_ID,
                                                       epsilon=0
                                                      )
    refine_sample_return_nll_loss, _ = mask_and_calculate_nll_loss(refine_logits,
                                                       refine_sample_returns,
                                                       config.CLS_ID,
                                                       epsilon=0
                                                      )
    # log probability of the actions(all the tokens in the vocab)
    sample_return_nll_loss = tf.stack([draft_sample_return_nll_loss, refine_sample_return_nll_loss], axis=0)    
    bert_f1_score = calculate_bert_f1(target_ids, candidate_returns)
    combined_bert_f1_score = tf.reduce_mean(bert_f1_score)
    # reshape by the actual batch size .i.e reshape the way it was before tiling it by 4
    bert_f1_score = tf.reshape(bert_f1_score, (actual_batch_size, 4))
    # avg across the batch
    bert_f1_score = tf.reduce_mean(bert_f1_score, axis=0)
    #(2,), (2,)
    sample_bert_f1, greedy_baseline_bert_f1 = tf.split(bert_f1_score, num_or_size_splits=2, axis=0)
    #(2,)
    pg_loss_with_baseline = (sample_bert_f1 - greedy_baseline_bert_f1)*sample_return_nll_loss
    #(2,)
    loss_with_pg = (1-gamma)*nll_loss + (gamma * pg_loss_with_baseline)
    loss_with_pg = tf.reduce_sum(loss_with_pg)

    return (loss_with_pg, combined_bert_f1_score)


def loss_function(target_ids, 
                 draft_logits, 
                 refine_logits,
                 candidate_returns, 
                 sample_returns):

    draft_loss, _ = mask_and_calculate_nll_loss(
                                         draft_logits,
                                         target_ids[:, 1:],
                                         config.SEP_ID
                                         )
    refine_loss, refine_target = mask_and_calculate_nll_loss(
                                                  refine_logits,
                                                  target_ids[:, :-1],
                                                  config.CLS_ID
                                                  )
    if not config.gamma == 0:
        loss = tf.stack([draft_loss, refine_loss], axis=0)
        target_ids = tf.tile(target_ids[:, 1:], [4, 1])
        loss, bert_f1_score  = calculate_policy_gradient_loss(
                                                              draft_logits,
                                                              refine_logits,
                                                              sample_returns, 
                                                              target_ids,
                                                              candidate_returns, 
                                                              loss
                                                              )
    else:
        loss = tf.reduce_sum([draft_loss, refine_loss])
        
        if config.show_BERT_F1_during_training:
            predicted = tf.math.argmax(refine_logits, axis=-1, output_type=tf.int64)
            bert_f1_score = calculate_bert_f1(target_ids[:, :-1], predicted)
            bert_f1_score = tf.reduce_mean(bert_f1_score)
        else:
            bert_f1_score = 0.0
        

    return (loss, bert_f1_score)
            
def get_optimizer():

    learning_rate = CustomSchedule(config.d_model)  
    optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=learning_rate, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 epsilon=1e-9
                                 )  

    return optimizer
