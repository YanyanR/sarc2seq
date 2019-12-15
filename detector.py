"""
This is a sarcasm detector that takes in a senentence,
and outputs a confidence score (the probability of being sarcastic).
"""
import tensorflow as tf
import numpy as np
from preprocess import get_data
import math
import os
import sklearn.model_selection as sk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

class Model(tf.keras.Model):
    """
    architecture: similar to the sentiment classifier
    embedding + LSTM + attention + LSTM + Dense
    """
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = 40
        self.embedding_size = 128
        self.batch_size = 100
        self.lstm1_size = 200 # based on sentiment classifier
        self.lstm2_size = 150 # based on sentiment classifier
        self.dense_size = 2 # based on sentiment classifier
        self.learning_rate = 0.01

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.lstm1 = tf.keras.layers.LSTM(self.lstm1_size, return_sequences=True, return_state=True)
        # TODO how to use this? attention_activation='sigmoid', return_attention=True
        self.att = tf.keras.layers.Attention()
        self.lstm2 = tf.keras.layers.LSTM(self.lstm2_size, return_sequences=True, return_state=True)
        self.D = tf.keras.layers.Dense(self.dense_size, activation='sigmoid')

    def call(self, inputs):
        embeddings = self.E(inputs)
        lstm1_output, lstm1_state1, lstm1_state2 = self.lstm1(embeddings)
        att_output = self.att([lstm1_output, lstm1_output])
        lstm2_output, lstm2_state1, lstm2_state2 = self.lstm2(att_output)
        temp = tf.keras.layers.GlobalAveragePooling1D()(lstm2_output)
        final_output = self.D(temp)
        # final_output = self.D(lstm2_output)

        return final_output

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    for i in range(0, len(train_inputs), model.batch_size):
        curr_inputs = np.array(train_inputs[i:i+model.batch_size])
        curr_labels = train_labels[i:i+model.batch_size]

        # backprop
        with tf.GradientTape() as tape:
            predictions = model.call(curr_inputs)
            loss = model.loss(predictions, curr_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 500 == 0:
            print("train acc: ", model.accuracy(predictions, curr_labels))


def test(model, test_inputs, test_labels):
    total_accuracy = []

    for i in range(0, len(test_inputs), model.batch_size):
        curr_inputs = np.array(test_inputs[i:i+model.batch_size])
        curr_labels = test_labels[i:i+model.batch_size]

        predictions = model.call(curr_inputs)
        total_accuracy.append(model.accuracy(predictions, curr_labels))

        if i % 500 == 0:
            print("test acc: ", sum(total_accuracy) / len(total_accuracy))

    return sum(total_accuracy) / len(total_accuracy)

def main():
    """
    dataset:
        S as positive examples.
        P and N as negative examples.

        S is a corpus containing sarcastic sentences (S). For this,
        sentences with sarcasm-positve gold labels are extracted from
        Ghosh et. al, 2016, and Riloff et. al, 2013

        P and N contain strong sentiment sentences from different data
        sources like : Stanford Treebank Dataset, IMDB Reviews
        Dataset, Amazon Product Reviews and Sentiment 140.
    """

    pos_fp = 'data/sentiment/P.txt'
    neg_fp = 'data/sentiment/N.txt'
    sarc_fp = 'data/sentiment/S.txt'
    discriminator_inputs_fp = 'discriminator_inputs.csv'
    discriminator_labels_fp = 'discriminator_labels.csv'

    print("Preprocessing data...")

    # if os.path.exists(discriminator_inputs_fp) and os.path.exists(discriminator_labels_fp):
    if False:
        inputs = loadtxt(discriminator_inputs_fp, delimiter=',')
        labels = loadtxt(discriminator_labels_fp, delimiter=',')
    else:
        pos_vec, neg_vec, sarc_vec, vocab, pad_token_idx = get_data(pos_fp, neg_fp, sarc_fp)

        normal_sentences = np.concatenate((pos_vec, neg_vec))
        sarc_labels = np.ones(sarc_vec.shape[0])
        normal_labels = np.zeros(normal_sentences.shape[0])

        inputs = np.concatenate((normal_sentences, sarc_vec))
        labels = np.concatenate((normal_labels, sarc_labels))
        inputs = asarray(inputs)
        labels = asarray(labels)
        savetxt(discriminator_inputs_fp, inputs, delimiter=',')
        savetxt(discriminator_labels_fp, labels, delimiter=',')

    train_x, test_x, train_y, test_y = sk.train_test_split(inputs, labels, test_size=0.2, random_state = 42)
    print("Finished processing.")
    model = Model(len(vocab))

    # Set-up the training step
    print("Start training...")
    train(model, train_x, train_y)
    print("Finished training")

    # Set up the testing steps
    print("Testing...")
    accuracy = test(model, test_x, test_y)
    print("final accuracy: ", accuracy)

if __name__ == '__main__':
    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)

    device = 'GPU:0' if gpu_available else 'CPU:0'
    try:
        with tf.device('/device:' + device):
            main()
    except RuntimeError as e:
        print(e)
