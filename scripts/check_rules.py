import tensorflow as tf
from transformers import AutoTokenizer

def set_memory_growth():
    # Set GPU memory growth
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if not gpu_devices:
        print("GPU not available so Running in CPU")
    else:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

def create_tokenizer(config, tokenizer_type=None,
                    source_tokenizer_path=None, 
                    target_tokenizer_path=None):
    
    if config.use_custom_tokenizer:
        available_tokenizers = {'BertWordPieceTokenizer': BertWordPieceTokenizer,
                                'ByteLevelBPETokenizer': ByteLevelBPETokenizer,
                                'CharBPETokenizer': CharBPETokenizer,
                                'SentencePieceBPETokenizer': SentencePieceBPETokenizer
                                }
        assert tokenizer_type not in available_tokenizers, (
            f'tokenizer_type should be either one in {available_tokenizers.keys()}'
                                )
        assert not ((available_tokenizers[tokenizer_type] == 'BertWordPieceTokenizer')
            and (config.target_language == 'ta'), ('Please donot use wordpiece\
                                                    for tamil try BPE')
                   )
        try:
            source_tokenizer = available_tokenizers[tokenizer_type](
                                                    f'.\\{source_tokenizer_path}-vocab.json', 
                                                    f'.\\{source_tokenizer_path}-merges.txt'
                                                    )
            target_tokenizer = available_tokenizers[tokenizer_type](
                                                f'.\\{target_tokenizer_path}-vocab.json', 
                                                f'.\\{target_tokenizer_path}-merges.txt'
                                                ) if config['task'] == 'translate' else source_tokenizer
        except Exception as e:
            print(e)
    else:
        source_tokenizer = AutoTokenizer.from_pretrained(config['input_pretrained_model'])
        target_tokenizer = AutoTokenizer.from_pretrained(config['target_pretrained_model'],
                            ) if config['task'] == 'translate' else source_tokenizer

    return(config, source_tokenizer, target_tokenizer)

def set_inference_rules(config):

    if config['draft_decoder_type'] == 'greedy':
        config['beam_size'] = 1
        config['top_p'] = 1 
        config['top_k'] = 0

    elif config['draft_decoder_type'] == 'only_beam_search':
        config['top_p'] = 1 
        config['top_k'] = 0 

    if config['refine_decoder_type'] == 'greedy':
        config['top_p'] = 1 
        config['top_k'] = 0

    return config

def load_and_set_bias(path):

    read_tensor = tf.io.read_file(path)
    output_bias_tensor = tf.io.parse_tensor(read_tensor, tf.float32)

    return tf.keras.initializers.Constant(output_bias_tensor.numpy())

def set_testing_rules(config):

    if config['unit_test_dataset_batch_size'] < config['gradient_accumulation_steps']:
        config['gradient_accumulation_steps'] =  config['unit_test_dataset_batch_size']
    if config['unit_test_dataset_batch_size'] > config['samples_to_test']:
        config['unit_test_dataset_batch_size'] = config['samples_to_test']
    if config['steps_to_print_training_info'] > config['unit_test_dataset_batch_size']:
        config['steps_to_print_training_info'] = config['unit_test_dataset_batch_size']
    config['grad_clipnorm'] = None
    config['run_tensorboard'] = False
    config['dropout_rate'] = config['epsilon_ls'] = config['l2_norm'] = 0
    config['samples_to_train'] = config['samples_to_test']

    return config

def set_training_rules(config):

    config['save_initial_weights'] = False
    config['run_init_eval'] = False
    config['init_loss_check'] = False
    config['input_independent_baseline_check'] = False
    config['check_model_capacity'] = False
    config['random_results_check'] = False
    config['print_config'] = True
    config['clear_log'] = False

    return config

def assert_config_values(config):

    available_draft_decoder_types = ['topktopp','greedy', 'only_beam_search']
    available_refine_decoder_types = ['greedy', 'topktopp']
    summarization_datasets = ['cnn_dailymail']
    translate_datasets = ['en_tam_parallel_text']
    implemented_tasks = ['summarize', 'translate']
    assert config.task in implemented_tasks, 'summarize and translate are implemented currently'
    assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
    assert config.eval_after_steps % config.steps_to_print_training_info == 0, (
    'For printing the training results without any issues "eval_after_steps"\
     must be a multiple of steps_to_print_training_info')
    assert config.steps_to_print_training_info >= config.gradient_accumulation_steps, (
    'To prevent undesirable training results please set gradient_accumulation_steps lesser\
    than steps_to_print_training_info')
    assert config.draft_decoder_type  in available_draft_decoder_types, (
            f'available draft decoder types are {available_draft_decoder_types}')
    assert config.refine_decoder_type in available_refine_decoder_types, (
        f'available refine decoder types are {available_refine_decoder_types}')
    if config.task == 'summarize':
        assert config.tfds_name in summarization_datasets, (
                f'{config.tfds_name} not currently added to summarize dataset list')
        assert config.input_seq_length > config.target_seq_length, (
            'input_seq_length must be greater than target_seq_length for summarize')
        assert config.input_pretrained_model == config.target_pretrained_model, (
                f'For {config.task} the input and target models must be same  for {config.task}')
    elif config.task == 'translate':
        assert config.tfds_name in translate_datasets , (
                f'{config.tfds_name} not currently added to translate dataset list')
        assert config.input_pretrained_model != config.target_pretrained_model, (
            f'For translate the input and target pre-trained BERT must not be same')

    return config

def check_and_assert_config(config):

    config, source_tokenizer, target_tokenizer = create_tokenizer(config)
    config['bert_score_model'] = config['target_pretrained_model'] if config['task'] == 'translate' else config['input_pretrained_model']
    config['input_vocab_size'] = source_tokenizer.vocab_size
    config['target_vocab_size'] = target_tokenizer.vocab_size
    # Special Tokens 
    config['PAD_ID']  = target_tokenizer.pad_token_id
    config['CLS_ID']  = target_tokenizer.cls_token_id
    config['MASK_ID'] = target_tokenizer.mask_token_id
    config['SEP_ID']  = target_tokenizer.sep_token_id
    if config['accumulate_gradients'] == False:
        config['gradient_accumulation_steps'] = 1

    if config.add_bias is not None:
        if config.task == 'translate':
            assert config.target_language in config.serialized_tensor_path, (
            'serialized Bias file not found,\
            please create it using helper scripts/create_bias script')
            config['add_bias'] = load_and_set_bias(config['serialized_tensor_path'])
        else:
            assert False,'add_bias is only available for translate task'
    if config['test_script']:
        config = set_testing_rules(config)
    else:
        config = set_training_rules(config)

    config = set_inference_rules(config)
    config = assert_config_values(config)
    
    return (config, source_tokenizer, target_tokenizer)

#set GPU memory growth
_ = set_memory_growth()