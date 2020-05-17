def calculate_policy_gradient_loss(greedy_op, 
                          sample_targets, 
                          target_ids):
    
    # tf.print()
    # tf.print('target_ids')
    # tf.print(tf.shape(target_ids))
    # tf.print()
    #with tf.device("CPU:0"):
    #sample_targets :- (batch_size, seq_len)
    #greedy_op :- (batch_size, seq_len)
    #target_ids :- (batch_size, seq_len)
    _, b_score_greedy_op = tf_write_output_sequence(target_ids, 
                                                    greedy_op, 
                                                    input_ids=False,step=1, 
                                                    write_output_seq=False, 
                                                    return_only_bert_score=True
                                                    )
    _, b_score_sampled_target = tf_write_output_sequence(target_ids, 
                                                         sample_targets, 
                                                         input_ids=False, step=2, 
                                                         write_output_seq=False, 
                                                         return_only_bert_score=True
                                                         )
    sampled_targets_3D = label_smoothing(tf.one_hot(sample_targets, depth=config.target_vocab_size))
    greedy_output_3D = tf.one_hot(greedy_op, depth=config.target_vocab_size)
    pg_loss = loss_object(sampled_targets_3D, greedy_output_3D)
    pg_mask = tf.math.logical_not(tf.math.equal(sample_targets, config.PAD_ID))
    pg_loss = mask_and_calculate_loss(pg_mask, pg_loss)
    pg_loss_with_baseline = (b_score_greedy_op-b_score_sampled_target)*pg_loss
    # tf.print('pg_loss_with_baseline')
    # tf.print()
    # tf.print(pg_loss_with_baseline)
    # tf.print(tf.shape(pg_loss_with_baseline))
    return pg_loss_with_baseline