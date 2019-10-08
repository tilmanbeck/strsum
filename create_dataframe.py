# coding:utf-8

import gzip
import json
import re

import pandas as pd
import tensorflow as tf
from nltk.tokenize import word_tokenize

import _pickle as cPickle

def remove_gold_arguments(df, gold_args):
    regex = re.compile("[^a-zA-Z0-9]")
    df["sentence_clean"] = df['sentence'].apply(lambda x: " ".join(regex.sub(" ", x).lower().strip().split()))
    indices = []
    for arg in gold_args:
        indices += df.index[
            df["sentence_clean"].str.contains(" ".join(regex.sub(" ", arg).lower().strip().split()), regex=False,
                                              case=False)].tolist()
    df.drop(index=indices, inplace=True)
    return df

def get_df(path):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)
        
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_df_direct(path):

    df = pd.read_csv(path + '/emnlp2018.tsv', sep='\t', header=0)
    df.drop_duplicates(inplace=True)

    with open(path + '/gold_arguments.json', 'r') as fp:
        gold_args = json.load(fp)

    gold_arguments = []
    for t in gold_args.keys():
        gold_arguments += [i['title'] for i in gold_args[t]['pro_points']]
        gold_arguments += [i['title'] for i in gold_args[t]['contra_points']]
    df = remove_gold_arguments(df, gold_arguments)
    
    tmp = []
    topics = pd.unique(df['topic'])
    for t in topics:
        # pro arguments as one review text, gold arguments as summmarx
        data = df[(df['topic'] == t) & (df['label'] == 'pro')]
        g = [i['title'] for i in gold_args[t]['pro_points']]
        summary = ".".join(g)
        summary = summary.replace("..", ".")
        reviewText = " ".join(data['sentence'].tolist())
        dic = {'reviewText': reviewText, 'summary': summary}
        tmp.append(dic)
        # con arguments as one review text, gold arguments as summmarx
        data = df[(df['topic'] == t) & (df['label'] == 'con')]
        g = [i['title'] for i in gold_args[t]['contra_points']]
        summary = ".".join(g)
        summary = summary.replace("..", ".")
        reviewText = " ".join(data['sentence'].tolist())
        dic = {'reviewText': reviewText, 'summary': summary}
        tmp.append(dic)
    return pd.DataFrame(tmp)


def get_tokens(doc):
    shortened = {
    '\'m': ' am',
    '\'re': ' are',
    '\'ll': ' will',
    '\'ve': ' have',
    'it\'s': 'it is',
    'isn\'t': 'is not',
    'aren\'t': 'are not',
    'wasn\'t': 'was　not',
    'weren\'t': 'were　not',
    'don\'t': 'do　not',
    'doesn\'t': 'does　not',
    'didn\'t': 'did　not',
    'haven\'t': 'have　not',
    'hasn\'t': 'has　not',
    'hadn\'t': 'had　not',
    'can\'t': 'can　not',
    'couldn\'t': 'could　not',
    'won\'t': 'will　not',
    'wouldn\'t': 'would　not',
    'cannot': 'can　not',
    'wanna': 'want to',
    'gonna': 'going to',
    'gotta': 'got to',
    'hafta': 'have to',
    'needa': 'need to',
    'outta': 'out of',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'lotta': 'lot of',
    'lemme': 'let me',
    'gimme': 'give me',
    'getcha': 'get you',
    'gotcha': 'got you',
    'letcha': 'let you',
    'betcha': 'bet you',
    'shoulda': 'should have',
    'coulda': 'could have',
    'woulda': 'would have',
    'musta': 'must have',
    'mighta': 'might have',
    'dunno': 'do not know',
    }
    
    doc = doc.lower()
    shortened_re = re.compile('(?:' + '|'.join(map(lambda x: '\\b' + x + '\\b', shortened.keys())) + ')')
    doc = shortened_re.sub(lambda x: shortened[x.group(0)], doc)
    
    doc = re.sub(r"\(.*?\)", "",doc)
    doc = re.sub(r"!", ".",doc)
    sents = [re.sub(r"[^A-Za-z0-9()\'\`_/]", " ", sent).lstrip() for sent in doc.split('.') if sent != '']
    
    tokens = []
    for s in sents:
        s = ' '.join(word_tokenize(s))
        s = s.replace(" n't ", "n 't ")
        s = s.split()
        if len(s) > 1: tokens.append(s)
    return tokens

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/')
    parser.add_argument('--output_path', default='data/arguments.pkl')
    parser.add_argument('--min_doc_l_train', default=10, type=int)
    parser.add_argument('--max_doc_l_train', default=60, type=int)
    parser.add_argument('--max_sent_l_train', default=50, type=int)
    parser.add_argument('--min_doc_l_test', default=5, type=int)
    parser.add_argument('--max_doc_l_test', default=60, type=int)
    parser.add_argument('--max_sent_l_test', default=50, type=int)

    args = parser.parse_args()
    print('parsing raw data...')
    #raw_review_df = get_df(args.input_path)
    raw_review_df = get_df_direct(args.input_path)
    print(raw_review_df)
    review_df = raw_review_df[(raw_review_df['reviewText'] != '') & (raw_review_df['summary'] != '')]
    
    print('splitting text into tokens...')
    review_df['tokens'] = review_df['reviewText'].apply(lambda d: get_tokens(d))
    review_df = review_df[(review_df['tokens'].apply(lambda x: len(x)) > 0)]
    
    review_df['doc_l'] = review_df['tokens'].apply(lambda d: len(d))
    review_df['max_sent_l'] = review_df['tokens'].apply(lambda d: max([len(s) for s in d]))
    review_df['summary_tokens'] = review_df['summary'].apply(lambda s: word_tokenize(s.lower()))
    review_df = review_df[(review_df['summary_tokens'].apply(lambda x: len(x)) > 0)]
    
    print('splitting data into train, dev, test...')
    topics = pd.unique(review_df['topic'])
    train_topics = topics[0:4]
    dev_topics = topics[4:6]
    test_topics = topics[6:]
    test_all_df = review_df[review_df['topic'].isin(test_topics)]
    dev_all_df = review_df[review_df['topic'].isin(dev_topics)]
    train_all_df = review_df[review_df['topic'].isin(train_topics)]
    
    train_df = train_all_df[(train_all_df['doc_l']>=args.min_doc_l_train)&(train_all_df['doc_l']<=args.max_doc_l_train)&(train_all_df['max_sent_l']<=args.max_sent_l_train)]
    dev_df = dev_all_df[(dev_all_df['doc_l']>=args.min_doc_l_test)&(dev_all_df['doc_l']<=args.max_doc_l_test)&(dev_all_df['max_sent_l']<=args.max_sent_l_test)]
    test_df = test_all_df[(test_all_df['doc_l']>=args.min_doc_l_test)&(test_all_df['doc_l']<=args.max_doc_l_test)&(test_all_df['max_sent_l']<=args.max_sent_l_test)]
    
    print('saving set of train, dev, test...')
    cPickle.dump((train_df, dev_df, test_df), open(args.output_path, 'wb'))
    
if __name__ == "__main__":
    main()
