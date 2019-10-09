import os

import tensorflow as tf

from train import train
from evaluate import evaluate
from data_structure import load_data
from attrdict import AttrDict


# special tokens
PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_vec_path', default='data/crawl-300d-2M.vec')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--datadir', default='data/', help='directory of input data')
    parser.add_argument('--dataname', default='input.pkl', help='name of data')
    parser.add_argument('--modeldir', default='model', help='directory of model')
    parser.add_argument('--modelname', default='args', help='name of model')

    parser.add_argument('--discourserank', default=True, help='flag of discourserank')
    parser.add_argument('--damp', default=0.9, type=float, help='damping factor of discourserank')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--log_period', default=500, type=int, help='valid period')

    parser.add_argument('--opt', default='Adagrad', help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--norm', default=1e-4, type=float, help='norm')
    parser.add_argument('--grad_clip', default=10.0, type=float, help='grad_clip')
    parser.add_argument('--keep_prob', default=0.95, type=float, help='keep_prob')
    parser.add_argument('--beam_width', default=10, type=int, help='beam_width')
    parser.add_argument('--length_penalty_weight', default=0.0, type=float, help='length_penalty_weight')

    parser.add_argument('--dim_hidden', default=256, type=int, help='hidden dimensions')
    parser.add_argument('--dim_str', default=128, type=int, help='dimension str')
    parser.add_argument('--dim_sent', default=384, type=int, help='dim_sent')

    # for evaluation
    parser.add_argument('--refdir', default='ref', help='refdir')
    parser.add_argument('--outdir', default='out', help='outdir')
    config = AttrDict(vars(parser.parse_args()))

    print('loading data...')
    train_batches, dev_batches, test_batches, embedding_matrix, vocab, word_to_id  = load_data(config)
    config.PAD_IDX = word_to_id[PAD]
    config.UNK_IDX = word_to_id[UNK]
    config.BOS_IDX = word_to_id[BOS]
    config.EOS_IDX = word_to_id[EOS]
    
    n_embed, d_embed = embedding_matrix.shape
    config.n_embed = n_embed
    config.d_embed = d_embed

    maximum_iterations = max([max([d._max_sent_len(None) for d in batch]) for ct, batch in dev_batches])
    config.maximum_iterations = maximum_iterations
    print('max iter:', maximum_iterations)
    
    if config.mode == 'train':
        train(config, train_batches, dev_batches, test_batches, embedding_matrix, vocab)
    elif config.mode == 'eval':
        evaluate(config, test_batches, vocab)

if __name__ == "__main__":
    run()
    
    
    
