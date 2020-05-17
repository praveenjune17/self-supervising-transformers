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
cos_sim_loss = tf.keras.losses.CosineSimilarity(reduction='none')
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
        except Exception as e:
            log.warning(f'{e} :- so setting ROUGE_avg to zero')
            avg_rouge_f1 = 0

        return avg_rouge_f1

    def evaluate_bert_score(self):
        
        try:
            _, _, bert_f1 = b_score(self.ref_sents, self.hyp_sents, 
                                  model_type=config.bert_score_model,
                                  batch_size=config.train_batch_size)
                                  #device='cpu')
            avg_bert_f1 = np.mean(bert_f1.numpy())
        except Exception as e:
            log.warning(f'{e} :- so setting BERT-score to zero')
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
        except Exception as e:
            log.warning(f'{e} :- so setting BLEU to zero')
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

    mask = tf.cast(mask, dtype=loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss

def compute_cosine_similarity(target_embeddings, 
                              generated_embeddings,
                              target_ids):

    cosine_sim_loss = cos_sim_loss(target_embeddings, generated_embeddings)
    #mask = tf.cast(tf.math.logical_not(tf.math.equal(target_ids, 0)), dtype=cosine_sim_loss.dtype)
    mask = tf.math.logical_not(tf.math.logical_or(tf.math.equal(
                                                            target_ids, 
                                                            config.CLS_ID
                                                                      ), 
                                                         tf.math.equal(
                                                            target_ids, 
                                                            config.PAD_ID
                                                                      )
                                                         )
                                      )
    cosine_sim_loss = mask_and_calculate_loss(mask, cosine_sim_loss)
    return cosine_sim_loss

def calculate_policy_gradient_loss(greedy_op, 
                          sample_targets, 
                          target_ids,
                          target_embeddings,
                          greedy_op_embeddings,
                          sample_return_embeddings):
    
    #sample_targets :- (batch_size, seq_len)
    #greedy_op :- (batch_size, seq_len)
    #target_ids :- (batch_size, seq_len+1)
    cosine_sim_greedy = compute_cosine_similarity(target_embeddings,
                                                  greedy_op_embeddings,
                                                  target_ids[:, :-1])
    
    cosine_sim_sampled_return = compute_cosine_similarity(target_embeddings,
                                                  sample_return_embeddings,
                                                  target_ids[:, :-1])
    
    sampled_targets_3D = label_smoothing(tf.one_hot(sample_targets, depth=config.target_vocab_size))
    greedy_output_3D = tf.one_hot(greedy_op, depth=config.target_vocab_size)
    pg_loss = loss_object(sampled_targets_3D, greedy_output_3D)
    pg_mask = tf.math.logical_not(tf.math.equal(sample_targets, config.PAD_ID))
    pg_loss = mask_and_calculate_loss(pg_mask, pg_loss)
    pg_loss_with_baseline = (cosine_sim_greedy-cosine_sim_sampled_return)*pg_loss

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

def loss_function(target_ids, draft_predictions, refine_predictions, 
                  draft_greedy_op, draft_sample_return,
                  refine_greedy_op, refined_sample_return, 
                  target_embeddings, draft_greedy_op_embeddings,
                  draft_sample_return_embeddings, refine_greedy_op_embeddings,
                  refined_sample_return_embeddings,model):

    # target_ids :- (batch, tar_seq_len+1)
    # draft_predictions :- (batch, tar_seq_len, vocab_size)
    # refine_predictions :- (batch, tar_seq_len, vocab_size)
    gamma = config.gamma
    target_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size))
    draft_loss, target = calculate_draft_loss(target_ids_3D, target_ids, draft_predictions)
    pg_draft_loss = calculate_policy_gradient_loss(draft_greedy_op, 
                                                   draft_sample_return, 
                                                   target_ids, target_embeddings,
                                                   draft_greedy_op_embeddings,
                        draft_sample_return_embeddings) if draft_greedy_op  is  not None else 0.0
    draft_loss = (1-gamma)*draft_loss + gamma * pg_draft_loss
    if refine_predictions is not None:
        refine_loss, target = calculate_refine_loss(target_ids_3D, target_ids, refine_predictions)
        pg_refine_loss = calculate_policy_gradient_loss(refine_greedy_op,
                                                        refined_sample_return, 
                                                        target_ids,target_embeddings,
                                                        refine_greedy_op_embeddings,
                        refined_sample_return_embeddings) if refine_greedy_op  is  not None else 0.0
        refine_loss = (1-gamma)*refine_loss + gamma * pg_refine_loss
    else:
        refine_loss = 0.0
    regularization_loss = tf.add_n(model.losses)
    total_loss = tf.reduce_sum([draft_loss, refine_loss, regularization_loss])

    return (total_loss, target) 
    
def get_loss_and_accuracy():

    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')

    return(loss, accuracy)
    
def write_output_sequence(true_target_ids, predictions, step, write_output_seq, input_ids, return_only_bert_score):
  
    ref_sents = target_tokenizer.batch_decode(true_target_ids, skip_special_tokens=True)
    hyp_sents = target_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    evaluate  = evaluation_metrics(ref_sents, hyp_sents)
    bert_f1  = evaluate.evaluate_bert_score()
    if write_output_seq:
        inp_sents = source_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy().decode('utf-8')), 'a') as f:
            for source, ref, hyp in zip(inp_sents, ref_sents, hyp_sents):
                f.write(source+'\t'+ref+'\t'+hyp+'\n')
    task_score = evaluate.evaluate_task_score() if return_only_bert_score is None else 0
    return (task_score, bert_f1)
  
  
def tf_write_output_sequence(tar_real, predictions, step, write_output_seq, input_ids, return_only_bert_score):

    return tf.py_function(write_output_sequence, 
                          [tar_real, predictions, step, write_output_seq, input_ids, return_only_bert_score], 
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
