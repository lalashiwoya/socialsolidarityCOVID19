# socialsolidarityCOVID19

PyTorch implementation of the methods, proposed in D. Grunow, D. Liu, S. Eger, A. Ils, 
**Changes in European solidarity before and during COVID-19:Evidence from a large-scale Crowd- and Expert-Annotated Twitterdataset**, *ACL-IJCNLP 2021*.

## Getting Started

### Dependencies
* Python3
* [transformers](https://github.com/huggingface/transformers)
* [pytorch-transformers](https://pypi.org/project/pytorch-transformers)

### Dataset
* You can find necessary datasets [here](https://drive.google.com/drive/folders/1X88L67fGOfTrWhI0UMKFu4dX5cYhAnof?usp=sharing).
* Also you can crawl tweets with the functions in the folder "Crawler".

## Usage
- An example of further pre_training using masked language modeling task and next sentence prediction task
```python
# train_corpus: path where saves the trained corpus in text file
# output_dir: directory to save the output
# do_lower_case:  whether lowercase text before tokenization
# epochs_to_generate: number of epochs to train for
# max_seq_len: maxinum sequence length

>>> python pregenerate_training_data.py --train_corpus  tweets_LM_6k.txt  --output_dir training_6k/ --do_lower_case --epochs_to_generate 20 --max_seq_len 150

# pregenerated_data: directory where saves the output of pregenerate_training_data.py 
# train_batch_size: batch size for training
# do_lower_case: whether lowercase text before tokenization
# epochs: number of epochs to train for

>>> python pre_training_mlm.py --pregenerated_data training_6k/   --train_batch_size 16  --do_lower_case --output_dir fine_tune/finetuned_lm_6k/ --epochs 20
```
- An example of further pre_training using sentiment classification
```python
# model: you can choose a model name from [bert, xlm]
# weights: you can choose pretrained weights from huggingface transformers ('bert-base-multilingual-cased'or 'xlm-roberta-base'), or self-trained weights
# optional:  
# --data_path: path of the data for sentiment classification  
# --output_dir: directory to save the model 

>>> python pre_training_sentiment_classification.py --model_type bert --pretrained_weights bert-base-multilingual-cased
```
- An example of training
```python
# model_type: you can choose a model from [bert, xlm]
# pretrained_weights: you can choose a pretrained weights from huggingface transformers ('bert-base-multilingual-cased' or 'xlm-roberta-base'), or self-trained weights
# optional:
# --model_path：path where saves the model  
# --oversample_from_train: whether do oversampling from training data  
# --translation: whether add translated data for training  
# --auto_data: whether add auto-labeled data for training  
# --is_norm: whether add batch normalization after the last hidden layer of bert or xlm model

>>> python train.py --model_type xlm --pretrained_weights xlm-roberta-base --translation --auto_data 
``` 
- An example of predicting
```python
# model_dir：directory where saves models  
# optional:
# --model_name：name of the model,for single model prediction  
# --data_dir： directory where saves tweets to be predicted 
# --output_dir： directory to save the prediction results
# --num_labels： the number of classes 
# --do_lower_case： whether lowercase text before tokenization

>>> python predict.py --model_dir saved_weights --model_name xlm_pytorch_model.bin --data_dir twitter_data --do_lower_case
``` 



