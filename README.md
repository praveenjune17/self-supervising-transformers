  Tensorflow 2.1

  ## Project features   
  a)2-stage transformer that can perform Machine translation or Text summarization.All the components of the architecture described in the [Base paper](https://arxiv.org/pdf/1902.09243v2.pdf) is implemented.    
  Summary:-  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)BERT or any pretrained language representation model can be used as Encoder  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)stage-1 decoder(intialized with pretrained model's weights) is trained using Autoregressive objective  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)stage-2 decoder(intialized with pretrained model's weights) is trained using MLM objective  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)Parameters are shared between the two decoders  
  b)Beam-search and topk_topp filtering could be used to generate text.[Inspired from](https://huggingface.co/blog/how-to-generate)  
  c)Used Huggingface's Transformers library for tokenization and to extract encoder embeddings from pretrained models  
  d)Gradient accumulation and Mixed precision policy can be enabled during training  
  e)BERT-F1 score Implemented in tensorflow which could be used as an evaluation metric  
  f)Use the trained model to visualize the embeddings of the source and target sentences in the valdiation dataset using helper_scripts/visualize_sentence_embeddings.py  
  g)For translation, create bias file (when using the bertified transformer) by running /helper_scripts/create_bias_for_multilingual.py this could help reduce the initial loss and hence saves some GPU cycles. Refer [init_well section in this link](http://karpathy.github.io/2019/04/25/recipe/)

  ## Note  
    a) Based on the experiments with BERT-F1-score in translation and summarization, BERT-F1-score of even 53 (or 0.53) doesn't mean that the   predicted sentences are related to the context of the input sentences. Test few samples with bert-score to get an idea of how it works.  
    b) The target pretrained model should have the same hidden size and embedding size i.e (d_model == embedding size) just like BERT.  
    c) Performance is slow during training  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)when using Tensorflow 2.2 on windows.  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)when show_BERT_F1_during_training is enabled  
    d)Expected source code changes when  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)gradient accumulation is implemented in optimizers   
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)Adafactor optimizer is implemented  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)Any new pretrained multilingual model is out  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#)Label smoothing for crossentropy fix is out  

  ## Instructions to run the model  
  To Train:- set the file paths, task, hyper_parameters, training_parms in the configuration.py and run train.py  
  To Infer:- set the checkpoint path in config.infer_ckpt_path and run inference_scripts/generate.py
