import tensorflow as tf
from tensorflow.keras.initializers import Constant
from transformers import TFAutoModel
from transformer import Decoder
from utilities import log
from configuration import config
from model_utils import (tile_and_mask_diagonal, create_masks, topp_topk,
                         with_column, mask_timestamp, draft_decoder)

def _embedding_from_bert():

    with tf.device("CPU:0"):  
        input_pretrained_bert = TFAutoModel.from_pretrained(
                                              config.input_pretrained_model, 
                                              trainable=False, 
                                              name=config.input_pretrained_model
                                              )
        target_pretrained_bert = TFAutoModel.from_pretrained(
                                    config.target_pretrained_model, 
                                    trainable=False, 
                                    name=config.target_pretrained_model
                                    ) if config['task'] == 'translate' else input_pretrained_bert
    decoder_embedding = target_pretrained_bert.get_weights()[0]
    log.info(f"Decoder_Embedding matrix shape '{decoder_embedding.shape}'")

    return (decoder_embedding, input_pretrained_bert, target_pretrained_bert)

class Bertified_transformer(tf.keras.Model):

    def __init__(
                  self, 
                  num_layers, 
                  d_model, 
                  num_heads, 
                  dff, 
                  input_vocab_size, 
                  target_vocab_size,
                  rate=config.dropout_rate, 
                  add_pointer_generator=None):
        super(Bertified_transformer, self).__init__()

        self.target_vocab_size = tf.constant(target_vocab_size)
        (decoder_embedding, self.encoder, 
        self.decoder_bert_model) = _embedding_from_bert()
        self.decoder_embedding = tf.keras.layers.Embedding(
                                       target_vocab_size, 
                                       d_model, 
                                       trainable=False,
                                       embeddings_initializer=Constant(decoder_embedding),
                                       name='Decoder-embedding'
                                       )
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate, 
                               add_pointer_generator=add_pointer_generator)
    def draft_summary(self,
                      input_ids,
                      enc_output,
                      look_ahead_mask,
                      padding_mask,
                      target_ids,
                      training):

        target_embeddings = self.decoder_embedding(target_ids)
        # draft_logits:-         (batch_size, tar_seq_len, tar_vocab_size)   
        # draft_attention_dist:- (batch_size, tar_seq_len, inp_seq_len)        
        draft_logits, draft_attention_dist = self.decoder(
                                                          input_ids,
                                                          target_embeddings, 
                                                          enc_output, 
                                                          training, 
                                                          look_ahead_mask, 
                                                          padding_mask
                                                          )
        # (batch_size, tar_seq_len, tar_vocab_size)
        return draft_logits, draft_attention_dist

    def fit(self, input_ids, 
            target_ids, training,
            look_ahead_mask, dec_padding_mask):
        
        enc_output = self.encoder(input_ids)[0]
        # (batch_size, seq_len, vocab_len), _
        draft_logits, draft_attention_dist = self.draft_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                look_ahead_mask=look_ahead_mask,
                                                                padding_mask=dec_padding_mask,
                                                                target_ids=target_ids,
                                                                training=training
                                                               )
        
        candidate_returns, sample_returns = (None, None)

        return (draft_logits, draft_logits, draft_attention_dist, 
                draft_attention_dist, candidate_returns, sample_returns)
        
    def call(self, input_ids, target_ids, encoding_padding_mask, dec_padding_mask, 
                 look_ahead_mask, training,
                 decoder_type=config.draft_decoder_type,
                 beam_size=config.beam_size,
                 length_penalty=config.length_penalty,
                 temperature=config.softmax_temperature, 
                 top_p=config.top_p,
                 top_k=config.top_k):

        #batch_size = tf.shape(input_ids)[0]
        #if training is not None:
        return self.fit(input_ids, target_ids, training, 
                        look_ahead_mask, dec_padding_mask)