from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset
import pandas as pd
import os
import datetime
import re


def get_datetime_from_string(datetime_string, new_format=""):
    """
    convert Twitter API date format to python datetime format
    """
    dt_obj = datetime.datetime.strptime(datetime_string, '%a %b %d %H:%M:%S %z %Y')
    if new_format == "":
        return dt_obj

    return dt_obj.strftime(new_format)


def build_dataset(path):
    """
    build train-dev-test sets from the original annotation files, and write them to csv. file
    :param path: path of the original annotation files
    :return:
    """
    texts = []
    labels = []
    date = []
    id = []
    files = [f for f in os.listdir(path) if f.endswith('.csv')]  # load all the csv. file names in the given path
    for file in files:
        data = pd.read_csv(os.path.join(path,file))

        for i in range(len(data)):
            if data['Agreement2'][i] != '?':  # if label is not uncertain
                texts.append(data['text'][i])
                labels.append(int(data['Agreement2'][i]) - 1)  # labels 1-4 -> integer 0-3
                date.append(get_datetime_from_string(str(data['date'][i])))
                id.append(data['id'][i])

    df = pd.DataFrame({"id": id, "date": date, "text": texts, "label": labels})

    # train test split
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    print(len(df), len(df_train), len(df_test))
    # train dev split
    df_dev = df_train.sample(frac=0.2, random_state=50)
    df_train = df_train.drop(df_dev.index).reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    # print(len(df_train), len(df_dev))
    # print(df_dev.head())
    # print(df_test.head())
    # print(df_train.loc[:, 'label'].value_counts())
    # print(df_dev.loc[:, 'label'].value_counts())
    # print(df_test.loc[:, 'label'].value_counts())
    df_train.to_csv('data/train.csv', index=False)
    df_dev.to_csv('data/dev.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)


def merge(label):
    """merge ambivalent and non-applicable labels (2, 3 -> 2)"""
    if label <= 2:
        return label
    else:
        return 2


def data_loader(path, do_merge=False):
    """
    function that reads tweet texts and labels from a given path
    :param path: path of a csv. file
    :param do_merge: whether merge ambivalent and non-applicable classes
    :return: a dataframe contains desired data
    """
    data = pd.read_csv(path)

    if do_merge:
        data['label'] = data['label'].apply(lambda x: merge(x))

    return data[['text', 'label']]


def load_translation(path, do_merge=False):
    """ function that reads translated texts and the corresponding labels """
    data = pd.read_csv(path)
    if do_merge:
        data['label'] = data['label'].apply(lambda x: merge(x))
    data = data[['translation', 'label']]
    # print(data.head())
    data = data.rename(columns={'translation': 'text'})  # rename column
    # print(data.head())
    return data


def oversampling(df, df_trans=None):
    """
    function that randomly selects samples from minority classes until reaching the same number with the majority class
    :param df: original dataset
    :param df_trans: translated data. If it's not None, sample from it. Otherwise, sample from the original dataset
    :return:
    """
    max_size = df['label'].value_counts().max()  # the number of items in the majority class
    ls = [df]
    if df_trans is not None:
        print('sample from translation...')
        for class_index, group in df_trans.groupby('label'):
            ls.append(group.sample(max_size - df['label'].value_counts()[class_index], replace=True, random_state=42))
    else:
        print('sample from the original dataset...')
        for class_index, group in df.groupby('label'):
            ls.append(group.sample(max_size - len(group), replace=True, random_state=42))
    # concat the original data and all the selected samples, so all the classes have the same number of items
    df_new = pd.concat(ls)
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df_new


def convert_data_into_features(data,tokenizer,max_len,batch_size,shuffle=True):
    dataset=CustomDataset(data, tokenizer, max_len)
    data_loader= DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def tweet_cleaner(tweet):
    """
    function that replaces mentions, hashtags and urls with special tokens
    """
    hashtags = re.compile(r"^#\S+|\s#\S+")
    mentions = re.compile(r"^@\S+|\s@\S+")
    urls = re.compile(r"https?://\S+")
    res = hashtags.sub(' #hashtag', tweet)
    res = mentions.sub(' @user', res)
    res = urls.sub(' url', res)
    res = ' '.join(res.split())
    return res.strip()


def build_cls_dataloader(path, tokenizer,max_len,batch_size):
    # Import the csv into pandas dataframe and add the headers
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(lambda x: tweet_cleaner(x))
    print(df)
    # dataset = df.sample(n=2000, replace=False, random_state=200)
    dataset = df
    # train_dev = dataset.sample(frac=0.8, random_state=200)
    # df_test = dataset.drop(train_dev.index).reset_index(drop=True)
    # train_dev = train_dev.reset_index(drop=True)

    df_train = dataset.sample(frac=0.8, random_state=200)
    df_dev = dataset.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    print(len(df_train), len(df_dev))
    print('class distribution of training set:')
    print(df_train.loc[:, 'label'].value_counts())
    print('class distribution of validation set:')
    print(df_dev.loc[:, 'label'].value_counts())
    train_loader=convert_data_into_features(df_train,tokenizer,max_len,batch_size)
    dev_loader=convert_data_into_features(df_dev,tokenizer,max_len,batch_size)
    return train_loader, dev_loader


def build_dataloader(src_path, do_merge, tokenizer, max_len, batch_size,
                     oversample_from_train=False, oversample_from_trans=False, translation=False, auto_data=False):
    # Load dataset
    df_train = data_loader(os.path.join(src_path, 'train.csv'), do_merge)

    # add auto-labeled data
    if auto_data:
        print("add auto data ...")
        df_extra = data_loader(os.path.join(src_path, 'auto_labeled_data.csv'))
        df_train = pd.concat([df_train, df_extra])
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # add translation
    df_trans = load_translation(os.path.join(src_path, 'train_translated.csv'), do_merge=do_merge)  # translated data
    if translation:
        print("add translated data ...")
        df_train = pd.concat([df_train, df_trans])
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # oversampling
    if oversample_from_train:
        df_train = oversampling(df_train)  # sample from original data
    elif oversample_from_trans:
        df_train = oversampling(df_train, df_trans)  # sample from translated data

    df_dev = data_loader(os.path.join(src_path, 'dev.csv'), do_merge)
    df_test = data_loader(os.path.join(src_path, 'test.csv'), do_merge)

    print(len(df_train), len(df_dev), len(df_test))
    print(df_train.loc[:, 'label'].value_counts())
    # print(df_dev.loc[:, 'label'].value_counts())
    # print(df_test.loc[:, 'label'].value_counts())
    # print(df_train.head())

    train_loader=convert_data_into_features(df_train,tokenizer,max_len,batch_size)
    dev_loader=convert_data_into_features(df_dev,tokenizer,max_len,batch_size)
    test_loader = convert_data_into_features(df_test, tokenizer, max_len, batch_size)

    return train_loader, dev_loader, test_loader