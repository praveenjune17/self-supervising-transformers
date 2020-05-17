# -*- coding: utf-8 -*-
import tempfile
import tensorflow as tf
import numpy as np
from rouge import Rouge
from bert_score import score as b_score
from official.nlp.transformer import compute_bleu
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log

loss_object = tf.keras.losses.CategoricalCrossentropy(
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


def label_smoothing(inputs, epsilon=config.epsilon_ls):
    # V <- number of channels
    V = inputs.get_shape().as_list()[-1] 
    epsilon = tf.cast(epsilon, dtype=inputs.dtype)
    V = tf.cast(V, dtype=inputs.dtype)

    return ((1-epsilon) * inputs) + (epsilon / V)

def mask_and_calculate_loss(mask, loss):

    mask   = tf.cast(mask, dtype=loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss

def policy_gradient_objective(greedy_op, sample_targets, 
                              target_ids):
    
    b_score_greedy_op = tf_write_output_sequence(target_ids, greedy_op)
    b_score_sampled_target = tf_write_output_sequence(target_ids, sample_targets)
    pg_loss = loss_object(sample_targets, greedy_op)
    pg_loss_with_baseline = (b_score_greedy_op-b_score_sampled_target)*pg_loss

    return pg_loss_with_baseline

def calculate_draft_loss(target_ids_3D, target_ids_2D, draft_predictions):
    
    draft_loss  = loss_object(target_ids_3D[:, 1:, :], draft_predictions)
    draft_mask = tf.math.logical_not(tf.math.equal(target_ids_2D[:, 1:], config.PAD_ID))
    draft_loss = mask_and_calculate_loss(draft_mask, draft_loss)
    draft_target = target_ids_3D[:, 1:, :]
    
    return (draft_loss, draft_target)

def calculate_refine_loss(target_ids_3D, target_ids_2D, refine_predictions):

    refine_loss  = loss_object(target_ids_3D[:, :-1, :], refine_predictions)
    refine_mask = tf.math.logical_not(tf.math.logical_or(tf.math.equal(
                                                            target_ids_2D[:, :-1], 
                                                            config.CLS_ID
                                                                      ), 
                                                         tf.math.equal(
                                                            target_ids_2D[:, :-1], 
                                                            config.PAD_ID
                                                                      )
                                                         )
                                      )
    refine_loss = mask_and_calculate_loss(refine_mask, refine_loss)
    refine_target = target_ids_3D[:, :-1, :]

    return (refine_loss, refine_target)

def loss_function(target_ids, draft_predictions, refine_predictions, model):

    # target_ids :- (batch, tar_seq_len+1)
    # draft_predictions :- (batch, tar_seq_len, vocab_size)
    # refine_predictions :- (batch, tar_seq_len, vocab_size)
    target_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size))
    draft_loss, target = calculate_draft_loss(target_ids_3D, target_ids, draft_predictions)
    if refine_predictions is not None:
        refine_loss, target = calculate_refine_loss(target_ids_3D, target_ids, refine_predictions)
    else:
        refine_loss = 0.0
    regularization_loss = tf.add_n(model.losses)
    total_loss = tf.reduce_sum([draft_loss, refine_loss, regularization_loss])

    return (total_loss, target) 
    
def get_loss_and_accuracy():

    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')

    return(loss, accuracy)
    
def write_output_sequence(true_target_ids, predictions, step, write_output_seq, input_ids=None):

    ref_sents = target_tokenizer.batch_decode(true_target_ids, skip_special_tokens=True)
    hyp_sents = target_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    evaluate  = evaluation_metrics(ref_sents, hyp_sents)
    task_score = evaluate.evaluate_task_score()
    bert_f1  = evaluate.evaluate_bert_score()
    if write_output_seq and (input_ids is not None):
        inp_sents = source_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy().decode('utf-8')), 'a') as f:
            for source, ref, hyp in zip(inp_sents, ref_sents, hyp_sents):
                f.write(source+'\t'+ref+'\t'+hyp+'\n')

    return (task_score, bert_f1)
  
  
def tf_write_output_sequence(tar_real, predictions, step=None, write_output_seq=None, input_ids=None):

    return tf.py_function(write_output_sequence, 
                          [tar_real, predictions, step, write_output_seq, input_ids], 
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
