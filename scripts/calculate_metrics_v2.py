# -*- coding: utf-8 -*-
import tempfile
import tensorflow as tf
import numpy as np
from rouge import Rouge
from bert_score import score as b_score
from official.nlp.transformer import compute_bleu
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log

negative_log_liklihood = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
cosine_similarity = tf.keras.losses.CosineSimilarity(reduction='none')#axis=1, reduction=tf.keras.losses.Reduction.NONE

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
def mask_and_calculate_nll_loss(predicted_ids, 
                            target_ids, 
                            mask_a_with,
                            mask_b_with=config.PAD_ID,
                            epsilon=config.epsilon_ls
                            ):

    
    target_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size), epsilon)
    loss = negative_log_liklihood(target_ids_3D, predicted_ids)
    mask = create_mask_for_pg(target_ids, loss, mask_a_with, mask_b_with)
    loss = loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss, target_ids_3D

def create_mask_for_pg(ids, scores, mask_a_with=config.CLS_ID, 
                      mask_b_with=config.PAD_ID):
      
    mask = tf.math.logical_not(tf.math.logical_or(
                                  tf.math.equal(ids, mask_a_with), 
                                  tf.math.equal(ids, mask_b_with))
                               )
    mask = tf.cast(mask, dtype=scores.dtype)

    return mask

def calculate_bert_f1(target_ids, predicted, scores):
      
    target_mask = create_mask_for_pg(target_ids, scores)
    predicted_return_mask = create_mask_for_pg(predicted, scores)
    mask = tf.matmul(target_mask[:, :, tf.newaxis,], predicted_return_mask[:, tf.newaxis,: ])
    mask = tf.transpose(mask, (0, 2, 1))
    scores = scores*mask
    recall = tf.reduce_max(scores, 1)
    precision = tf.reduce_max(scores, 2)
    recall = tf.reduce_sum(recall, 1)
    precision = tf.reduce_sum(precision, 1)
    recall = recall/tf.reduce_sum(target_mask, -1)
    precision = precision/tf.reduce_sum(predicted_return_mask, -1)
    f1_score = (2*(precision*recall))/(precision+recall)
    bert_f1 = tf.reduce_mean(f1_score)

    return bert_f1

def calculate_policy_gradient_loss(predictions, 
                          sample_returns, 
                          target_ids,
                          greedy_returns,
                          sample_returns_scores,
                          greedy_returns_scores,
                          nll_loss,
                          gamma=config.gamma
                          ):
    
    #sample_targets :- (batch_size, seq_len)
    #greedy_op :- (batch_size, seq_len)
    #target_ids :- (batch_size, seq_len+1)
    sample_return_nll_loss, _ = mask_and_calculate_nll_loss(predictions,
                                                       sample_returns,
                                                       config.CLS_ID,
                                                       epsilon=0
                                                      )
    sample_bert_f1 = calculate_bert_f1(target_ids, sample_returns, sample_returns_scores)
    greedy_baseline_bert_f1 = calculate_bert_f1(target_ids, greedy_returns, greedy_returns_scores)
    pg_loss_with_baseline = (sample_bert_f1 - greedy_baseline_bert_f1)*sample_return_nll_loss
    # tf.print('greedy_baseline_bert_f1')
    # tf.print(greedy_baseline_bert_f1)
    # tf.print('sample_bert_f1')
    # tf.print(sample_bert_f1)
    # tf.print()
    loss_with_pg = (1-gamma)*nll_loss + (gamma * pg_loss_with_baseline)

    return loss_with_pg


def loss_function(target_ids, draft_predictions, refine_predictions,
                  draft_sample_returns_scores, draft_greedy_returns_scores,
                 draft_sample_returns,
                 draft_greedy_returns,
                 refine_sample_returns_scores, refine_greedy_returns_scores,
                 refine_sample_returns,
                 refine_greedy_returns):

    
    draft_loss, _ = mask_and_calculate_nll_loss(
                                         draft_predictions,
                                         target_ids[:, 1:],
                                         config.SEP_ID
                                         )
    refine_loss, refine_target = mask_and_calculate_nll_loss(
                                                  refine_predictions,
                                                  target_ids[:, :-1],
                                                  config.CLS_ID
                                                  )
    draft_loss = calculate_policy_gradient_loss(draft_predictions, 
                                          draft_sample_returns, 
                                          target_ids[:, :-1],
                                          draft_greedy_returns,
                                          draft_sample_returns_scores,
                                          draft_greedy_returns_scores,
                                          draft_loss
                                          )
    refine_loss = calculate_policy_gradient_loss(refine_predictions, 
                                          refine_sample_returns, 
                                          target_ids[:, :-1],
                                          refine_greedy_returns,
                                          refine_sample_returns_scores,
                                          refine_greedy_returns_scores,
                                          refine_loss
                                          )
    total_loss = tf.reduce_sum([draft_loss, refine_loss])

    return (total_loss, refine_target)
    
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
