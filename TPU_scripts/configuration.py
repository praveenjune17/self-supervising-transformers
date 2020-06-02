# -*- coding: utf-8 -*-
import os
import platform
from bunch import Bunch
from check_rules import check_and_assert_config

unit_test = {
      'check_evaluation_pipeline' : False,
      'check_model_capacity' : False,
      'check_training_pipeline' : False,
      'check_predictions_shape' : False,
      'clear_log' : True,
      'detokenize_samples' : False,
      'gpu_memory_test' : False,
      'init_loss_check' : False,
      'input_independent_baseline_check' : False, 
      'print_config' : True,
      'random_results_check' : False,
      'samples_to_test' : 1,
      'save_initial_weights' : False,
      'test_script' : False,
      'unit_test_dataset_batch_size' : 1
          }
model_parms = {
     'add_bias' : False,               # set values as True|None Increases the inital bias of Tamil vocabs 
     'activation' : 'relu',
     'add_pointer_generator': True,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 1024,                      # feed forward network hidden parameters
     'input_pretrained_model': 'distilroberta-base',  #distilroberta-base, #bert-base-uncased , #google/electra-small-discriminator
     'input_seq_length': 50,
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'target_language' : 'ta',
     'target_pretrained_model' : 'distilbert-base-multilingual-cased',#'bert-base-uncased',
                                                                     #'bert-base-multilingual-cased',
                                                                    #'distilbert-base-multilingual-cased'
     'target_seq_length': 20,
     'task':'translate'            # must be translate or summarize
     }
training_parms = {
     'accumulate_gradients' : True,
     'display_model_summary' : True,
     'early_stop' : False,
     'enable_jit' : False,                    # disabled for windows automatically
     'eval_after_steps' : 10000,              # Evaluate after these many training steps
     'gamma' : 0.0,
     'gradient_accumulation_steps': 18,   
     'last_recorded_value':  None,
     'min_train_loss' : 1.0,
     'monitor_metric' : 'perplexity',      # perplexity or bert_f1_score
     'num_parallel_calls' : -1,
     'run_tensorboard': True,
     'samples_to_train' : -1,                  # -1 takes all the samples
     'samples_to_validate' : -1,
     'show_BERT_F1_during_training' : False,   # for performance reasons set this to False 
     'steps_to_print_training_info': 200,      # print training progress per number of batches specified
     'tfds_name' : 'en_tam_parallel_text',            #cnn_dailymail,en_tam_parallel_text     # tfds dataset to be used
     'init_tolerance' : 0,
     'tolerance_threshold': 7,          # Stop training after the threshold is reached
     'tokens_per_batch' : 4050,
     'use_custom_tokenizer' : None,
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'validate_when_train_loss_is' : 7.0           # run evaluation when loss reaches 10
     }
inference_decoder_parms = {
    'beam_size': 1,              
    'draft_decoder_type' : 'greedy',     # 'greedy', 'only_beam_search', 'topktopp' --> topktopp filtering + beam search
    'length_penalty' : 0.6,
    'refine_decoder_type' : 'greedy',     # 'greedy', 'topktopp' --> beam search not possible
    'softmax_temperature' : 1,
    'top_p' : 1, 
    'top_k' : 25                         
    }
h_parms = {
   'dropout_rate': 0.1,
   'epochs': 4,
   'epsilon_ls': 0.1,                  # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.0,
   'learning_rate': None,              # set None to create decayed learning rate schedule
   'train_batch_size': 1024,
   'validation_batch_size' : 16
   }                                    
dataset_name = training_parms['tfds_name']
model = 'bertified_transformer'
core_path = 'gs://tensorflow_en_tam_dataset/'#"/content/drive/My Drive/"
path_seperator = '\\' if platform.system() == 'Windows' else '/'
file_path = {
        'best_ckpt_path' : os.path.join(core_path, f"best_checkpoints{path_seperator}{dataset_name+'_'+model}{path_seperator}"),  
        'checkpoint_path' : os.path.join(core_path, f"checkpoints{path_seperator}{dataset_name+'_'+model}{path_seperator}"),
        'initial_weights' : os.path.join(core_path, f"initial_weights{path_seperator}{dataset_name+'_'+model}{path_seperator}"),
        'infer_csv_path' : None,
        'infer_ckpt_path' : 'D:\\Local_run\\checkpoints\\en_tam_parallel_text_bertified_transformer\\ckpt-301',
        'input_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_en"),
        'log_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}tensorflow.log"),
        'output_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_ta"),
        'output_sequence_write_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}summaries{path_seperator}"),
        'serialized_tensor_path' : os.path.join(core_path, 'saved_serialized_tensor_'+ model_parms['target_language']),
        'tensorboard_log' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}tensorboard_logs{path_seperator}"),
        'tfds_data_dir' : os.path.join(core_path, f'Tensorflow_datasets{path_seperator}{dataset_name}_dataset'),
        'tfds_data_version' : None,
        'tf_records_path' : os.path.join(core_path, f'Tensorflow_datasets{path_seperator}{dataset_name}_dataset'),
        'train_csv_path' : None
            }
config = Bunch(model_parms)
config.update(unit_test)
config.update(training_parms)
config.update(inference_decoder_parms)
config.update(h_parms)
config.update(file_path)
config, source_tokenizer, target_tokenizer = check_and_assert_config(config)
