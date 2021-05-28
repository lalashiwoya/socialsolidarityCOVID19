from utils.models import bertCNN, bertDPCNN, BertForSequenceClassification, XLMRobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel, AdamW
from utils.data_processor import build_cls_dataloader
from train import train, evaluate
from sklearn import metrics
from utils.pytorchtools import EarlyStopping
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

parser = ArgumentParser()
parser.add_argument("--model_type", type=str, required=True,
                    help="Choose model type from bert, xlm, bert_cnn, bert_dpcnn")
parser.add_argument("--pretrained_weights", type=str, required=True,
                    help="""Name of the pretrained weights from huggingface transformers 
                    (e.g. 'bert-base-multilingual-cased', 'xlm-roberta-base'), or path of self-trained weights""")
# parser.add_argument("--data_path", type=Path, required=True, help="Path of the data for sentiment classification")
# parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--data_path", type=str, default='data/sentiment_8k.csv', help="Path of the data for sentiment classification")
parser.add_argument("--output_dir", type=str, default='fine_tune', help="Directory to save the fine-tuned model")
parser.add_argument("--num_labels", type=int, default=2, help="The number of classes for sentiment classification")

parser.add_argument("--is_norm", action="store_true",
                    help="Whether add batch normalization after the last hidden layer of bert or xlm model")

parser.add_argument("--max_len", type=int, default=150, help="Maximal sequence length of bert or xlm model")
parser.add_argument("--epochs", type=int, default=20, help="Maximal number of epochs to train")
parser.add_argument("--patience", type=int, default=6, help="How many epochs to wait after last improvement (for early stopping).")
parser.add_argument("--batch_size", default=4, type=int, help="The batch size for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for AdamW.")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument('--do_lower_case', action="store_true", help="Whether lowercase text before tokenization")
args = parser.parse_args()

# fix random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# define some key variables for training
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
print(f'max_len={MAX_LEN}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, learning_rate={LEARNING_RATE}, seed={args.seed}, patience={args.patience}, do_lower_case={args.do_lower_case}')

model_path = os.path.join(args.output_dir, f"{args.model_type}_finetune_cls")
num_labels = args.num_labels

if args.model_type in ['bert', 'xlm']:
    if args.is_norm:
        if args.model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels,
                                                                  output_hidden_states=True)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels,
                                                                        output_hidden_states=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels)
elif args.model_type == 'bert_cnn':
    embed_model = AutoModel.from_pretrained(args.pretrained_weights)
    model = bertCNN(embed_model=embed_model, dropout=0.2, kernel_num=4, kernel_sizes=[3, 4, 5, 6],
                    num_labels=num_labels)
elif args.model_type == 'bert_dpcnn':
    embed_model = AutoModel.from_pretrained(args.pretrained_weights)
    model = bertDPCNN(embed_model=embed_model, num_filters=100, num_labels=num_labels)
else:
    raise ValueError(
        'Error model type! Model type must be one of the following 4 types: bert, xlm, bert_cnn or bert_dpcnn')


print(f'model_type={args.model_type}, pretrained_weights={args.pretrained_weights}, num_labels={num_labels}')

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, do_lower_case=args.do_lower_case)
dataloader_train, dataloader_dev = build_cls_dataloader(path=args.data_path,
                                                        tokenizer=tokenizer,
                                                        max_len=MAX_LEN,
                                                        batch_size=BATCH_SIZE)

model.to(device)
optimizer = AdamW(model.parameters(),
                  lr=LEARNING_RATE,
                  eps=1e-8)

early_stopping = EarlyStopping(patience=args.patience, verbose=True, monitor='val_f1', path=model_path)


for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}')
    train(dataloader_train, model, optimizer, is_norm=args.is_norm)
    predictions, true_vals = evaluate(dataloader_dev, model, is_norm=args.is_norm)
    val_f1 = metrics.f1_score(true_vals, predictions, average='macro')
    print(f'F1 score (macro) on dev set: {val_f1}')
    if args.model_type == 'bert':
        early_stopping(val_f1, model.bert)
    elif args.model_type == 'xlm':
        early_stopping(val_f1, model.roberta)
    else:
        early_stopping(val_f1, model.embed)

    if early_stopping.early_stop:
        print("Early stopping")
        break

embed_model = AutoModel.from_pretrained(args.pretrained_weights)
embed_model.load_state_dict(torch.load(model_path))
print('Saving model...')
model_to_save = embed_model.module if hasattr(embed_model, 'module') else embed_model  # Only save the model it-self
output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(args.output_dir)
print('Successfully saved model. Done.')
