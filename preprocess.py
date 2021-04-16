import utils
from collections import Counter
import random
import numpy as np


def preprocess(text):
    # get list of words
    words = utils.preprocess(text)

    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    ## Subsampling
    threshold = 1e-5
    word_counts = Counter(int_words)
    # print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

    total_count = len(int_words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    # discard some frequent words, according to the subsampling equation
    # create a new list of words for training
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    preprocessed = {'train_words': train_words,
                    'vocab_to_int': vocab_to_int,
                    'int_to_vocab': int_to_vocab,
                    'freqs': freqs}
    return preprocessed


# Get nearby elements for a given index
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)


# Get batches
def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y
