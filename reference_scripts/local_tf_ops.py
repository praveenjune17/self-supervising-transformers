import tensorflow as tf
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_dataset
from configuration import config
from utilities import (log, train_output_sequence_writer, 
                    valid_output_sequence_writer, detokenize)
from create_model import source_tokenizer, target_tokenizer, Model
from decode_utils import create_padding_mask
from calculate_metrics import (get_loss_and_accuracy, loss_function, 
                               optimizer, tf_write_output_sequence)


tf.config.optimizer.set_jit(config.enable_jit)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                      tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                     ]
  
model_metrics = 'Step {}\n,\
                 Train Loss {:.4f}\n,\
                 Train_Accuracy {:.4f}\n,\
                 ROUGE_f1 {:4f}\n,\
                 BERT_f1 {:4f}\n'
evaluation_step  = 'Time taken for {} step : {} secs' 
checkpoint_details = 'Saving checkpoint at step {} on {}'
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Train_Loss {:.4f} Train_Accuracy {:.4f}'

train_loss, train_accuracy = get_loss_and_accuracy()
gradient_accumulators = []

@tf.function(input_signature=train_step_signature)
def train_step(input_ids, 
               target_ids_, 
               target_ids, 
               draft_mask, 
               refine_mask,
               grad_accum_flag):
    with tf.GradientTape() as tape:
        (draft_predictions, draft_attention_weights, 
          refine_predictions, refine_attention_weights) = Model(
                                                               input_ids,  
                                                               target_ids_,
                                                               True
                                                               )
        train_variables = Model.trainable_variables
        draft_output_sequence_loss = loss_function(target_ids[:, 1:, :], 
                                                   draft_predictions, 
                                                   draft_mask
                                                   )
        if config.use_refine_decoder:
            refine_output_sequence_loss = loss_function(target_ids[:, :-1, :], 
                                                        refine_predictions, 
                                                        refine_mask
                                                        )
            predictions = refine_predictions
            target = target_ids_[:, :-1]
        else:
            refine_output_sequence_loss = 0
            predictions = draft_predictions
            target = target_ids_[:, 1:]
              
        regularization_loss = tf.add_n(Model.losses)
        loss = draft_output_sequence_loss + refine_output_sequence_loss + regularization_loss
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients  = tape.gradient(scaled_loss, train_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    # Initialize the shadow variables with same type as the gradients 
    if not gradient_accumulators:
        for tv in gradients:
          gradient_accumulators.append(tf.Variable(tf.zeros_like(tv), 
                                                   trainable=False)
                                      )
    # accmulate the gradients to the shadow variables
    for (accumulator, grad) in zip(gradient_accumulators, gradients):
        accumulator.assign_add(grad)
    # apply the gradients and reset them to zero if the flag is true
    if grad_accum_flag:
        optimizer.apply_gradients(zip(gradient_accumulators, train_variables))
        for accumulator in (gradient_accumulators):
            accumulator.assign(tf.zeros_like(accumulator))
        train_loss(loss)
        train_accuracy(target, predictions)
    return predictions

@tf.function(input_signature=val_step_signature)
def val_step(
             input_ids,
             target_ids_,
             step, 
             write_output_seq):
    dec_padding_mask = create_padding_mask(input_ids)
    (draft_predictions, _,  
     refine_predictions, _) = Model.predict( 
                                    input_ids,
                                    dec_padding_mask,
                                    False
                                    )
    
    if config.use_refine_decoder:
      predictions = refine_predictions
    else:
      predictions = draft_predictions
    rouge, bert = tf_write_output_sequence(target_ids_[:, 1:], 
                                           predictions[:, 1:], 
                                           step, 
                                           write_output_seq)  
    return (rouge, bert)

def evaluate_validation_set(
                           validation_dataset, 
                           step
                           ):
    rouge_score_total = 0
    bert_score_total = 0
    for (batch, (input_ids, target_ids_)) in enumerate(validation_dataset):
        # calculate rouge and bert score for only the first batch
        if batch == 0:
          rouge_score, bert_score = val_step(input_ids,
                                             target_ids_,  
                                             step, 
                                             config.write_batch1_predictions
                                             )
        else:
          rouge_score, bert_score  =  val_step(input_ids,
                                               target_ids_, 
                                               step, 
                                               False
                                               )
        rouge_score_total+=rouge_score
        bert_score_total+=bert_score
    return (rouge_score_total/(batch+1), 
            bert_score_total/(batch+1))

def eval_step(input_ids, 
               target_ids_, 
               target_ids, 
               draft_mask, 
               refine_mask
               ):
  
    (draft_predictions, draft_attention_weights, 
      refine_predictions, refine_attention_weights) = Model(
                                                            input_ids,  
                                                            target_ids_,
                                                            False
                                                            )
    draft_output_sequence_loss = loss_function(target_ids[:, 1:, :], 
                                               draft_predictions, 
                                               draft_mask)
    if config.use_refine_decoder:
        refine_output_sequence_loss = loss_function(target_ids[:, :-1, :], 
                                                    refine_predictions, 
                                                    refine_mask
                                                    )
    else:
      refine_output_sequence_loss = 0
    regularization_loss = tf.add_n(Model.losses)
    loss = draft_output_sequence_loss + refine_output_sequence_loss + regularization_loss
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
        log.info('No checkpoint found')
    return (ckpt_manager)

# run every batch
def batch_run_check(batch, start):
    if config.run_tensorboard:
        with train_output_sequence_writer.as_default():
          tf.summary.scalar('train_loss', train_loss.result(), step=batch)
          tf.summary.scalar('train_accuracy', train_accuracy.result(), step=batch)
    if config.display_model_summary:
        log.info(Model.summary())
        log.info(batch_zero.format(time.time()-start))
        config.display_model_summary = False
    log.info(
             batch_run_details.format(
                                     train_loss.result(), 
                                     train_accuracy.result()
                                     )
            )
    return train_loss.result()



def train_sanity_check(tokenizer, predictions, target_id):
    # use the last sample in the batch
    predicted, target = detokenize(tokenizer, 
                                   tf.squeeze(tf.argmax(predictions,axis=-1)[-1:]), 
                                   tf.squeeze(target_id[:, :-1][-1:])
                                   )
    log.info(f'the true output_sequence is {target}')
    log.info(f'the predicted output_seq with teacher forcing is \
      {        predicted if predicted else "empty hence evaluation will be skipped"}')
    return predicted

def training_results(
                          step, 
                          rouge_score, 
                          bert_score,
                          timing_info,
                          ckpt_save_path
                          ):

      log.info(
                model_metrics.format(
                                    step, 
                                    train_loss.result(), 
                                    train_accuracy.result(), 
                                    rouge_score, 
                                    bert_score
                                    )
              )
      log.info(evaluation_step.format(step, timing_info))
      log.info(checkpoint_details.format(step, ckpt_save_path))