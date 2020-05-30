Tensorflow 2.1

### Project features   
a)Implemented 2-stage transformer architecture that can perform Machine translation or Text summarization.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)[Base paper](https://arxiv.org/pdf/1902.09243v2.pdf)   
	#)stage-1 decoder is trained using Autoregressive objective  
	#)stage-2 decoder is trained using MLM objective  
	#)All the components of the architecture described in the paper is coded.  
b)Beam-search and topk_topp filtering could be used to generate text ,[Inspired from](https://huggingface.co/blog/how-to-generate)  
c)Used Huggingface's Transformers library for tokenization and to extract embeddings from pretrained models  
d)Gradient accumulation and Mixed precision policy enabled training  
e)BERT-F1 score Implemented in tensorflow which could be used as a evaluation metric  
f)Sanity_check.py script to test the components of the pipeline individually  
g)Use the trained model to visualize the embeddings of the source and target sentences in the valdiation dataset using helper_scripts/visualize_sentence_embeddings.py  
h)For translation, create bias file (when using the bertified transformer) by running /helper_scripts/create_bias_for_multilingual.py this could help reduce the initial loss and hence saves some GPU cycles. Refer [init_well section in this link](http://karpathy.github.io/2019/04/25/recipe/)

## Note  
  a) Based on the experiments with BERT-F1-score in translation and summarization, BERT-F1-score of even 53 (or 0.53) doesn't mean that the   predicted sentences are related to the context of the input sentences. Test few samples with bert-score to get an idea of how it works.  
  b) The target pretrained model should have the same hidden size and embedding size i.e (d_model == embedding size) just like BERT.  
  c) Performance is slow during training  
  		#)when using Tensorflow 2.2 on windows.  
  		#)when show_BERT_F1_during_training is enabled  
Expected source code changes when  
  a) gradient accumulation is implemented in optimizers by tensorflow community
  b) Adafactor optimizer is implemented  
  c) ELECTRA multilingual model is out  
  d) label smoothing for crossentropy fix is out  
To Train:- set the file paths, task, hyper_parameters, training_parms in the configuration.py and run train.py  
To Infer:- set the checkpoint path in config.infer_ckpt_path and run 
