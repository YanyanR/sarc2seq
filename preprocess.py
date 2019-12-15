import numpy as np
from collections import Counter
import enchant
import re
import os

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 30
ENGLISH_DICT = enchant.Dict("en_US")

def clean_text_file(filepath, dest):
    """
    @param file: a file path pointing to a text file
    @param dest: a file path to save the cleaned file
    @return None

    Removes all punctuation and words that are not english words from a
    text file and saves as a separate file.
    """
    text = []
    with open(filepath, 'r') as in_file, open(dest, 'w') as out_file:
        for line in in_file:
                line = line.lower()
                line = re.sub('[^a-zA-z0-9\s]', '', line)
                line = line.split()
                line = [w if ENGLISH_DICT.check(w) else UNK_TOKEN for w in line]

                out_file.write(" ".join(line))
                out_file.write("\n")

def read_data(filepath):
    """
    Extracts and returns sentences from text file

    @param file_name: the path to the txt file
    @return a list of sentences ready to be processed
    """
    text = []
    with open(filepath, 'r') as data_file:
        for line in data_file: text.append(line.split())
    return text

def pad_corpus(sentences):
    """
    Pads the sentences in each corpus to make them all of length WINDOW_SIZE

    @param sentences: list of sentences of variable length
    @return list of padded sentences of length WINDOW_SIZE
    """
    padded_sentences = []
    for sentence in sentences:
        padded_sentence = sentence[:WINDOW_SIZE-1] # < 30 elements
        padded_sentence += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - 1 - len(padded_sentence))
        padded_sentences.append(padded_sentence)

    return padded_sentences

def build_vocab(pos_sentences, neg_sentences, sarc_sentences):
    """
    Builds vocab from list of sentences

    @param sentences: list of padded sentences (lists of words and tokens)
    @return tuple (dict{word->index}, pad_token_idx)
    """
    tokens = []
    for s in pos_sentences: tokens.extend(s)
    for s in neg_sentences: tokens.extend(s)
    for s in sarc_sentences: tokens.extend(s)

    count_map = Counter(tokens)
    vocab_size = 20000
    total_words = sum(count_map.values())

    reduced_vocab = count_map.most_common(vocab_size)
    # print("portion words kept: {}, total words: {}".format(sum([s for (f,s) in reduced_vocab]) / total_words, total_words))
    # print("num pads: {}".format(count_map[PAD_TOKEN]))
    reduced_vocab = [first for (first, second) in reduced_vocab]

    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + reduced_vocab)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab, vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
    """
    Hash sentences to vocab indices

    @param vocab:  dict{word->idx}
    @param sentences:  list of padded sentences (lists of words and tokens)
    @return numpy array of integers, representing hashed sentences using vocab dict
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def get_data(pos_fp, neg_fp, sarc_fp):
    """
    Uses helper functions in file to read, parse, pad, and vectorize data

    @param pos_fp: Path to the positive sentiment training file.
    @param neg_fp: Path to the negative sentiment training file.
    @param sarc_fp: Path to the sarcastic training file.

    @return pos_vec: list of vectorizd sentences with positive sentiment
    @return neg_vec: list of vectorizd sentences with negative sentiment
    @return sarc_vec: list of vectorizd sarcastic sentences
    @return vocab: Dict containg word->index mapping
    @return pad_token_idx: the ID used for *PAD* in the English vocab
    """
    pos_sentences = read_data(pos_fp)
    neg_sentences = read_data(neg_fp)
    sarc_sentences = read_data(sarc_fp)

    pos_padded = pad_corpus(pos_sentences)
    neg_padded = pad_corpus(neg_sentences)
    sarc_padded = pad_corpus(sarc_sentences)

    vocab, pad_token_idx = build_vocab(pos_padded, neg_padded, sarc_padded)

    pos_vec = convert_to_id(vocab, pos_padded)
    neg_vec = convert_to_id(vocab, neg_padded)
    sarc_vec = convert_to_id(vocab, sarc_padded)

    return pos_vec, neg_vec, sarc_vec, vocab, pad_token_idx
