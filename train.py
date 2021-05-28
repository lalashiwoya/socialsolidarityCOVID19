from transformers import AutoTokenizer, BertModel
from utils.models import BertForSequenceClassification, XLMRobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AdamW
from utils.pytorchtools import EarlyStopping
from utils.models import bertCNN, bertDPCNN
from utils.data_processor import build_dataloader
from sklearn import metrics
from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def train(dataloader_train, model, optimizer, is_norm=False):
    model.train()

    loss_train_total = 0

    for _, data in enumerate(dataloader_train, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        if is_norm:
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets, is_norm=is_norm)
        else:
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    loss_train_avg = loss_train_total / len(dataloader_train)
    print(f'Training loss: {loss_train_avg}')


def evaluate(dataloader_val, model, is_norm=False):
    model.eval()
    predictions, true_vals = [], []

    for _, data in enumerate(dataloader_val, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        with torch.no_grad():
            if is_norm:
                outputs = model(ids, mask, token_type_ids, labels=targets, is_norm=is_norm)
            else:
                outputs = model(ids, mask, token_type_ids, labels=targets)
        big_val, big_idx = torch.max(outputs[1].data, dim=1)

        predictions.extend(big_idx.tolist())
        true_vals.extend(targets.tolist())

    return predictions, true_vals


def show_performance(truth, pred):
    precision = metrics.precision_score(truth, pred, average=None)
    recall = metrics.recall_score(truth, pred, average=None)
    f1 = metrics.f1_score(truth, pred, average=None)
    print(f'Precision:{precision}, Recall:{recall},  F1: {f1}')
    f1_score_micro = metrics.f1_score(truth, pred, average='micro')
    f1_score_macro = metrics.f1_score(truth, pred, average='macro')
    print(f"F1 Score (Micro) = {f1_score_micro}, F1 Score (Macro) = {f1_score_macro}")
    return f1_score_macro


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose model type from bert, xlm, bert_cnn, bert_dpcnn")
    parser.add_argument("--pretrained_weights", type=str, required=True,
                        help="""Name of the pretrained weights from huggingface transformers 
                        (e.g. 'bert-base-multilingual-uncased', 'xlm-roberta-base'), or path of self-trained weights""")
    parser.add_argument("--model_path", default=None, type=str,
                        help="path to save the model")

    parser.add_argument("--oversample_from_train", action="store_true",
                        help="Whether do oversampling from training data")
    parser.add_argument("--oversample_from_trans", action="store_true",
                        help="Whether do oversampling from translated data")
    parser.add_argument("--translation", action="store_true",
                        help="Whether add translated data for training")
    parser.add_argument("--auto_data", action="store_true",
                        help="Whether add auto-labeled data for training")
    parser.add_argument("--is_norm", action="store_true",
                        help="Whether add batch normalization after the last hidden layer of bert or xlm model")

    parser.add_argument('--do_merge', type=bool, default=True,
                        help="Whether merge ambivalent and non-applicable classes")
    parser.add_argument("--max_len", type=int, default=150, help="Maximal sequence length of bert or xlm model")
    parser.add_argument("--epochs", type=int, default=50, help="Maximal number of epochs to train")
    parser.add_argument("--patience", type=int, default=6,
                        help="How many epochs to wait after last improvement (for early stopping).")
    parser.add_argument("--batch_size", default=4, type=int, help="The batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--do_lower_case', action="store_true", help="Whether lowercase text before tokenization")

    args = parser.parse_args()

    # fix random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # define some key variables for training
    DO_MERGE = args.do_merge
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    print(
        f'max_len={MAX_LEN}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, learning_rate={LEARNING_RATE}, seed={args.seed}, patience={args.patience}, do_lower_case={args.do_lower_case}')

    if args.model_path is None:
        model_path = os.path.join('saved_weights', f'{args.model_type}_pytorch_model.bin')
    else:
        model_path = args.model_path

    num_labels = 3 if DO_MERGE else 4

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
        embed_model = BertModel.from_pretrained(args.pretrained_weights)
        model = bertCNN(embed_model=embed_model, dropout=0.2, kernel_num=4, kernel_sizes=[3, 4, 5, 6],
                            num_labels=num_labels)
    elif args.model_type == 'bert_dpcnn':
        embed_model = BertModel.from_pretrained(args.pretrained_weights)
        model = bertDPCNN(embed_model=embed_model, num_filters=100, num_labels=num_labels)
    else:
        raise ValueError(
            'Error model type! Model type must be one of the following 4 types: bert, xlm, bert_cnn or bert_dpcnn')

    print(f'model_type={args.model_type}, pretrained_weights={args.pretrained_weights}, num_labels={num_labels}')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, do_lower_case=args.do_lower_case)

    dataloader_train, dataloader_dev, dataloader_test = build_dataloader(src_path='data', do_merge=DO_MERGE,
                                                                         tokenizer=tokenizer, max_len=MAX_LEN,
                                                                         batch_size=BATCH_SIZE,
                                                                         oversample_from_train=args.oversample_from_train,
                                                                         oversample_from_trans=args.oversample_from_trans,
                                                                         translation=args.translation,
                                                                         auto_data=args.auto_data)

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
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # if val_f1 < early_stopping.best_score:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    predictions, true_vals = evaluate(dataloader_train, model, is_norm=args.is_norm)
    print('Train scores:', metrics.f1_score(true_vals, predictions, average=None),
          metrics.f1_score(true_vals, predictions, average='macro'))
    print('Evaluation....')
    predictions, true_vals = evaluate(dataloader_dev, model, is_norm=args.is_norm)
    f1_val = show_performance(true_vals, predictions)
    print('Testing....')
    predictions, true_vals = evaluate(dataloader_test, model, is_norm=args.is_norm)
    f1_test = show_performance(true_vals, predictions)



