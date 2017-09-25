import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    # W_norm = np.zeros(W.shape)
    # d = (np.sum(W ** 2, 1) ** (0.5))
    # W_norm = (W.T / d).T
    # evaluate_vectors(W_norm, vocab, ivocab)

    neigh = NearestNeighbors(metric='cosine')
    neigh.fit(W)

def top_10(word):
    if word not in vocab:
        return "error, value not found"
    dist, neighbors = neigh.kneighbors([W[vocab[word],]], 10)

    return zip([ivocab[idx] for idx in neighbors.flatten()], dist.flatten())

main()