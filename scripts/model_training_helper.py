import tensorflow as tf
import time
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from configuration import config
from utilities import log, create_tensorboard_parms
from create_model import Model
from model_utils import create_masks
from training_house_keeping import monitor_eval_metrics, training_results, train_sanity_check
from calculate_metrics import (loss_function, calculate_bert_f1, get_optimizer, mask_and_calculate_nll_loss)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.config.optimizer.set_jit(config.enable_jit)

(train_output_sequence_writer, _, _) = create_tensorboard_parms()
optimizer = get_optimizer()
train_loss = tf.keras.metrics.Mean(name='Train_Loss')
avg_bert_score = tf.keras.metrics.Mean(name='bert_f1_mean')
avg_perplexity = tf.keras.metrics.Mean(name='perplexity')
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Train_Loss {:.4f} BERT_f1_score {:.4f}' if config.show_BERT_F1_during_training else 'Train_Loss {:.4f}'
gradient_accumulators = []

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      #tf.TensorSpec(shape=(None), dtype=tf.string)
                     ]

@tf.function(input_signature=train_step_signature, experimental_compile=config.enable_jit)
def train_step(input_ids, 
               target_ids,
               grad_accum_flag):
    
    _, combined_mask, dec_padding_mask = create_masks(
                                                        input_ids, 
                                                        target_ids[:, :-1]
                                                        )
    with tf.GradientTape() as tape:
        (draft_logits, refine_logits, draft_attention_weights, 
          refine_attention_weights, 
          candidate_returns,  
          sample_returns) = Model(
                                   input_ids,
                                   dec_padding_mask=dec_padding_mask,
                                   target_ids=target_ids,
                                   look_ahead_mask=combined_mask, 
                                   training=True,
                                   )
        train_variables = Model.trainable_variables
        loss, bert_f1_score = loss_function(target_ids,
                                     draft_logits, 
                                     refine_logits,
                                     candidate_returns,
                                     sample_returns
                                     )
        regularization_loss = tf.add_n(Model.losses)
        total_loss = tf.reduce_sum([loss, regularization_loss])
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients  = tape.gradient(scaled_loss, train_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    if config.accumulate_gradients:
        # Initialize the shadow variables with same type as the gradients 
        if not gradient_accumulators:
            for tv in gradients:
              gradient_accumulators.append(tf.Variable(tf.zeros_like(tv), 
                                                       trainable=False))
        # accmulate the gradients to the shadow variables
        for (accumulator, grad) in zip(gradient_accumulators, gradients):
            accumulator.assign_add(grad)
        # apply the gradients and reset them to zero if the flag is true
        if grad_accum_flag:
            optimizer.apply_gradients(zip(gradient_accumulators, train_variables))
            for accumulator in (gradient_accumulators):
                accumulator.assign(tf.zeros_like(accumulator))
            train_loss(loss)
            
    else:
        optimizer.apply_gradients(zip(gradients, train_variables))
        train_loss(loss)

    return refine_logits, bert_f1_score

#@tf.function(input_signature=val_step_signature) #slow with tf.function
def val_step(
             input_ids,
             target_ids):

    (draft_predicted_ids, 
     draft_attention_weights,  
     refine_predicted_ids_2D, 
     refine_attention_weights,
     refine_logits) = Model( 
                           input_ids,
                           decoder_type=config.draft_decoder_type,
                           beam_size=config.beam_size,
                           length_penalty=config.length_penalty,
                           temperature=config.softmax_temperature, 
                           top_p=config.top_p,
                           top_k=config.top_k,
                           target_ids=None,
                           dec_padding_mask=None, 
                           look_ahead_mask=None, 
                           training=None
                           )
    # print(f'target_ids {tf.shape(target_ids)}')
    # print(f'refine_logits {tf.shape(refine_logits)}')
    refine_validation_loss, _ = mask_and_calculate_nll_loss(refine_logits,
                                                            target_ids,
                                                            config.PAD_ID,
                                                            epsilon=0,
                                                            )
    perplexity = tf.math.exp(refine_validation_loss)
    perplexity /= config.validation_batch_size
    bert_f1 = calculate_bert_f1(target_ids, refine_predicted_ids_2D)
    return (perplexity, bert_f1, 
            draft_attention_weights, 
            refine_attention_weights)

def evaluate_validation_set(
                           validation_dataset, 
                           decoder_type=config.draft_decoder_type,
                           beam_size=config.beam_size,
                           length_penalty=config.length_penalty,
                           temperature=config.softmax_temperature, 
                           top_p=config.top_p,
                           top_k=config.top_k
                           ):

    avg_bert_score.reset_states()
    avg_perplexity.reset_states()
    for (batch, (input_ids, target_ids)) in enumerate(validation_dataset, 1):
        # calculate rouge and bert score for only the first batch
        (perplexity,bert_f1,
        draft_attention_weights, 
        refine_attention_weights) =  val_step(input_ids,
                                             target_ids
                                             )
        
        
        avg_bert_score.update_state(bert_f1)
        avg_perplexity.update_state(perplexity)
    return (avg_perplexity.result().numpy(),
            avg_bert_score.result().numpy(),
            draft_attention_weights,
            refine_attention_weights
            )

def eval_step(input_ids, 
               target_ids, 
               ):

    target_inp = target_ids[:, :-1]
    _, combined_mask, dec_padding_mask = create_masks(input_ids, target_inp)  
    (draft_predictions, draft_attention_weights, 
    refine_predictions, refine_attention_weights) = Model(
                                                           input_ids,
                                                           dec_padding_mask=dec_padding_mask,
                                                           target_ids=target_inp,
                                                           look_ahead_mask=combined_mask, 
                                                           training=False
                                                           )
    loss, target = loss_function(target_ids, 
                         draft_predictions,
                         refine_predictions, 
                         Model
                         )
    train_loss(loss)
    log.info(Model.summary())
    if config.save_initial_weights:
        initial_weights = os.path.join(config.initial_weights,'initial_weights')
        Model.save_weights(initial_weights)

    return loss
    
def check_ckpt(checkpoint_path):

    ckpt = tf.train.Checkpoint(
                               Model=Model,
                               optimizer=optimizer
                              )
    ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                              checkpoint_path, 
                                              max_to_keep=10)
    if tf.train.latest_checkpoint(checkpoint_path):
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log.info(ckpt_manager.latest_checkpoint +' restored')
    else:
        log.warning('No checkpoint found so using the initialized_weights')

    return ckpt_manager

def batch_run_check(batch, start_time, bert_f1_score):

    if config.run_tensorboard:
        with train_output_sequence_writer.as_default():
          tf.summary.scalar('train_loss', train_loss.result(), step=batch)
    if config.display_model_summary:
        log.info(Model.summary())
        log.info(batch_zero.format(time.time()-start_time))
        config.display_model_summary = False
    log.info(
             batch_run_details.format(
                                     tf.debugging.assert_all_finite(
                                                 train_loss.result(), 
                                                 message="NaN's or Inf's.", 
                                                 name='NAN_assertion'
                                                ), 
                                     bert_f1_score.numpy()
                                     )
            )

def save_evaluate_monitor(ck_pt_mgr, val_dataset, 
            target_tokenizer, predictions, 
            target_ids, step, start_time, bert_f1_score
            ):

    ckpt_save_path = ck_pt_mgr.save()
    # print the detokenized training output of a single sample
    train_sanity_check(target_tokenizer, predictions, target_ids, log)
    # Run evaluation only if the train_loss is lesser than validate_when_train_loss_is
    if train_loss.result() < config.validate_when_train_loss_is:
        (validation_perplexity,
         val_bert_score, 
         draft_attention_weights,
         refine_attention_weights) = evaluate_validation_set(       
                                                      val_dataset
                                                      )
        early_stop_training = monitor_eval_metrics(
                              ckpt_save_path,
                              validation_perplexity, 
                              val_bert_score, 
                              train_loss.result(), 
                              step,
                              log,
                              config
                              )
    else:
        log.info('Not running evaluation since loss is not lesser than config.validate_when_train_loss_is')
        (validation_perplexity, val_bert_score) = (0, 0)
        early_stop_training = False
        draft_attention_weights = None
        refine_attention_weights = None
    
    training_results(
                      step,
                      train_loss.result(), 
                      bert_f1_score.numpy(),
                      validation_perplexity,
                      val_bert_score,
                      (time.time() - start_time),
                      ckpt_save_path,
                      log,
                      config
                      )
    train_loss.reset_states()
    return (early_stop_training,
            draft_attention_weights,
            refine_attention_weights)
