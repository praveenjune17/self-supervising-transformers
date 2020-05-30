import time
import re
import pickle
import tensorflow as tf
import sys
sys.path.insert(0, 'D:\\Local_run\\self-supervising-transformers\\scripts')
from profanity_check import predict_prob as vulgar_check
from create_model import  Model
from model_utils import create_padding_mask
from configuration import config, source_tokenizer, target_tokenizer


def preprocess(sentence):
    en_blacklist = '"#$%&\()*+-./:;<=>@[\\]^_`♪{|}~='
    cleantxt = re.compile('<.*?>')
    # Lower case english lines
    sentence_lower = sentence.lower()
    # Remove html tags from text
    cleaned_sentence = re.sub(cleantxt, '', sentence_lower)
    # Remove english text in tamil sentence and tamil text in english sentence
    cleaned_sentence = ''.join([ch for ch in cleaned_sentence if ch not in en_blacklist])
    # Remove duplicate empty spaces
    preprocessed_sentence = " ".join(cleaned_sentence.split())
    vulgar_prob = vulgar_check([preprocessed_sentence])[0]
    if vulgar_prob > 0.6:
        raise ValueError("No vulgar words please :) ")
    else:
        return preprocessed_sentence

def translate():
    
    en_input = input('Enter the sentence to be translated-> ')
    en_input = preprocess(en_input)
    input_ids = tf.constant(source_tokenizer.encode(en_input))[None, :]
    #tf.convert_to_tensor([[config.input_CLS_ID] + source_tokenizer.encode(en_input) + [config.input_SEP_ID]])
    #print(f'input_ids')
    #print(input_ids)
    dec_padding_mask = create_padding_mask(input_ids)

    start = time.time()
    (preds_draft_summary, _,
    preds_refine_summary, _) = Model.predict(input_ids,
                                               batch_size=1,
                                               draft_decoder_type='topktopp',
                                               beam_size=10,
                                               length_penalty=0.6,
                                               temperature=1, 
                                               top_p=0.9,
                                               top_k=25)
    translated_sequence = target_tokenizer.decode(tf.squeeze(preds_refine_summary), skip_special_tokens=True)
    print(f'Translated output--> {translated_sequence if translated_sequence else "EMPTY"}')
    print(f'Time to process --> {round(time.time()-start)} seconds')

if __name__ == '__main__':
    ckpt = tf.train.Checkpoint(
                               Model=Model
                              )
    ckpt.restore('D:\\Local_run\\checkpoints\\en_tam_parallel_text_bertified_transformer\\ckpt-301').expect_partial()
    translate()
