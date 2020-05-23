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
class evaluation_metrics:

    def __init__(self, true_output_sequences, predicted_output_sequences, task=config.task):
        self.ref_sents = true_output_sequences
        self.hyp_sents = predicted_output_sequences
        self.calculate_rouge = Rouge()
        self.task = task

    def evaluate_rouge(self):
        
        try:
            all_rouge_scores = self.calculate_rouge.get_scores(self.ref_sents , self.hyp_sents)
            avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                              rouge_scores['rouge-2']["f"], 
                              rouge_scores['rouge-l']["f"]]) for rouge_scores in all_rouge_scores])
        except:
            log.warning('Some problem while calculating ROUGE so setting it to zero')
            avg_rouge_f1 = 0

        return avg_rouge_f1

    def evaluate_bert_score(self):
        
        try:
            _, _, bert_f1 = b_score(self.ref_sents, self.hyp_sents, 
                                  model_type=config.bert_score_model,
                                  device='cpu')
            avg_bert_f1 = np.mean(bert_f1.numpy())
        except:
            log.warning('Some problem while calculating BERT score so setting it to zero')
            avg_bert_f1 = 0
            
        return avg_bert_f1

    def evaluate_bleu_score(self, case_sensitive=False):

        ref_filename = tempfile.NamedTemporaryFile(delete=False)
        hyp_filename = tempfile.NamedTemporaryFile(delete=False)

        with tf.io.gfile.GFile(ref_filename.name, 'w') as f_ref:
            with tf.io.gfile.GFile(hyp_filename.name, 'w') as f_hyp:
                for references, hypothesis_output in zip(self.ref_sents , self.hyp_sents):
                    f_hyp.write(hypothesis_output+'\n')
                    f_ref.write(references+'\n')
        try:
            bleu_score = compute_bleu.bleu_wrapper(ref_filename = ref_filename.name, 
                                                   hyp_filename = hyp_filename.name,
                                                   case_sensitive = False)
        except:
            log.warning('Some problem while calculating BLEU score so setting it to zero')
            bleu_score = 0

        return bleu_score

    def evaluate_task_score(self):

        return self.evaluate_bleu_score() if self.task=='translate' else self.evaluate_rouge()

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

# nll :- negative_log_liklihood
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
    # (4*batch_size, tar_seq_len, target_vocab_size)
    target_embeddings = Model.decoder_bert_model(target_ids, attention_mask=target_mask)[0]
    # (4*batch_size, cand_seq_len, target_vocab_size)
    predicted_embeddings = Model.decoder_bert_model(predicted, attention_mask=predicted_return_mask)[0]
    target_embeddings_normalized = target_embeddings/(tf.norm(target_embeddings, axis=-1)[:, :, tf.newaxis])
    predicted_embeddings_normalized = predicted_embeddings/(tf.norm(predicted_embeddings, axis=-1)[:, :, tf.newaxis])
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
    loss = tf.stack([draft_loss, refine_loss], axis=0)
    target_ids = tf.tile(target_ids[:, 1:], [4, 1])
    policy_gradients_loss, bert_f1_score  = calculate_policy_gradient_loss(
                                                            draft_logits,
                                                            refine_logits,
                                                            sample_returns, 
                                                            target_ids,
                                                            candidate_returns, 
                                                            loss
                                                            )

    return (policy_gradients_loss, bert_f1_score)
    
def get_loss_and_accuracy():

    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')

    return(loss, accuracy)
    
def write_output_sequence(true_target_ids, predictions, step, write_output_seq, input_ids):
  
    ref_sents = target_tokenizer.batch_decode(true_target_ids, skip_special_tokens=True)
    hyp_sents = target_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    evaluate  = evaluation_metrics(ref_sents, hyp_sents)
    bert_f1  = evaluate.evaluate_bert_score()
    if write_output_seq:
        inp_sents = source_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy().decode('utf-8')), 'a') as f:
            for source, ref, hyp in zip(inp_sents, ref_sents, hyp_sents):
                f.write(source+'\t'+ref+'\t'+hyp+'\n')
    task_score = evaluate.evaluate_task_score()

    return (task_score, bert_f1)
  
  
def tf_write_output_sequence(input_ids, tar_real, predictions, step, write_output_seq):

    return tf.py_function(write_output_sequence, 
                          [input_ids, tar_real, predictions, step, write_output_seq], 
                          Tout=[tf.float32, tf.float32]
                          )
    
def get_optimizer():

    learning_rate = config.learning_rate if config.learning_rate else CustomSchedule(config.d_model)    
    if config.grad_clipnorm:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=learning_rate, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 clipnorm=config.grad_clipnorm,
                                 epsilon=1e-9
                                 )
    else:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=learning_rate, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 epsilon=1e-9
                                 )

    return optimizer
