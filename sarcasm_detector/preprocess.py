import numpy as np
import tensorflow as tf
import numpy as np
import re
import os
import collections

PAD_TOKEN = "*PAD*"

def read_data(file_name, MAX_WINDOW_SIZE):
    text = []
    line_lengths = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file:
            line = line.lower()
            line = re.sub('[^a-zA-z0-9\s]', '', line)
            line = line.split()
            if (len(line) > MAX_WINDOW_SIZE):
                MAX_WINDOW_SIZE = len(line)
            line_lengths.append(len(line))
            text.append(line)

    # freq = collections.Counter(line_lengths)
    # print(freq)
    print(sum(l > 20 for l in line_lengths) / len(line_lengths), len(line_lengths))

    return text, MAX_WINDOW_SIZE

def get_data(S, P, N):
    MAX_WINDOW_SIZE = 0
    sarcasm, MAX_WINDOW_SIZE = read_data(S, MAX_WINDOW_SIZE)
    positive, MAX_WINDOW_SIZE = read_data(P, MAX_WINDOW_SIZE)
    negative, MAX_WINDOW_SIZE = read_data(N, MAX_WINDOW_SIZE)
    import pdb; pdb.set_trace()

    sarcasm_label = np.ones(len(sarcasm))
    normal_label = np.zeros(len(normal))

    inputs = sarcasm + normal
    labels = sarcasm_label + normal_label

    # shuffle the inputs and labels before returning
    indices = tf.random.shuffle(tf.range(int(len(labels))))
    inputs = tf.gather(inputs, indices)
    labels = tf.gather(labels, indices)

    splitting_point = round(len(labels)*0.8)
    train_x = inputs[:splitting_point]
    train_y = labels[:splitting_point]
    test_x = inputs[splitting_point:]
    test_y = labels[splitting_point:]

    vocab_size = 20000
    #vocab size was defined in the paper's source code

    return train_x, train_y, test_x, test_y, vocab_size
