'''
Reference script https://colab.research.google.com/drive/1IAIeaTdl4bJlWxDJS6AXFnBCg1B92E5Z#scrollTo=NsaUPscGZ-Mq
'''

def convert_examples_to_features(
                                examples,
                                input_tokenizer=input_tokenizer,
                                target_tokenizer=target_tokenizer,
                                input_max_length=512,
                                output_max_length=72,                                                                                    
                                label_list=None,
                                pad_token=0
                              ):
    
    features = []
    for (ex_index, (input_text, output_text) ) in enumerate(examples):
        
        example = InputExample(None, input_text.numpy().decode('utf-8'), output_text.numpy().decode('utf-8'), None)   
        input_ids = input_tokenizer.encode(example.text_a)
        output_ids = target_tokenizer.encode(example.text_b)      
        # Zero-pad up to the sequence length.
        input_padding_length = input_max_length - len(input_ids)
        output_padding_length = output_max_length - len(output_ids)
        
        input_ids = input_ids + ([pad_token] * input_padding_length)
        output_ids = output_ids + ([pad_token] * output_padding_length)

        features.append(
            InputFeatures(
                input_ids=input_ids, token_type_ids =output_ids
            )
        )

    def gen():
        for ex in features:
            yield (ex.input_ids,ex.token_type_ids)

    return tf.data.Dataset.from_generator(
        gen,
        (tf.int32,tf.int32),
        (tf.TensorShape([None]),tf.TensorShape([None]))
    )
