# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:\\Local_run\\models')
import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import numpy as np
from official.nlp.transformer import compute_bleu
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log
from rouge import Rouge

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

def write_output_sequence(input_ids, true_target_ids, predictions, filename='decoded_file', write_output_seq=True):

    ref_sents = []
    hyp_sents = []
    inp_sents = []
    for input_id, true_target_id, predicted_hyp in zip(input_ids, true_target_ids, predictions):
        detokenized_refs = target_tokenizer.decode(tf.squeeze(true_target_id), skip_special_tokens=True)
        detokenized_hyp_sents = target_tokenizer.decode(tf.squeeze(predicted_hyp), skip_special_tokens=True)
        detokenized_input_sequence = source_tokenizer.decode(tf.squeeze(input_id), skip_special_tokens=True)
        ref_sents.append(detokenized_refs)
        hyp_sents.append(detokenized_hyp_sents)
        inp_sents.append(detokenized_input_sequence)
    evaluate = evaluation_metrics(ref_sents, hyp_sents)
    task_score = evaluate.evaluate_task_score()
    bert_f1  = evaluate.evaluate_bert_score()
    if write_output_seq:
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy().decode('utf-8')), 'a') as f:
            for source, ref, hyp in zip(inp_sents, ref_sents, hyp_sents):
                f.write(source+'\t'+ref+'\t'+hyp+'\n')

    return (task_score, bert_f1)

if __name__ == '__main__':
  #Restore the model's checkpoints
  restore_chkpt(config.infer_ckpt_path)
  infer_dataset = infer_data_from_df()
  write_output_sequence(input_ids, true_target_ids, predictions)
