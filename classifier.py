import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Attention, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from preprocess import get_data
import math
import os
import sklearn.model_selection as sk

class SentimentClassifier(tf.keras.Model):
    """
    This model classifies an input sentence as containing positive or negative
    sentiment. After this model is trained, it will be used to extract the
    amount that each word in a sentence contributes to the overall sentiment,
    and will remove the words that contribute to the overall sentiment the most
    in order to neutralize the sentence.
    """
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = 40
        self.embedding_size = 128
        self.batch_size = 128

        self.lstm1_size = 200
        self.lstm2_size = 150
        self.dense_size = 2
        self.learning_rate = 0.01

        self.model.add(Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size))
        self.model.add(LSTM(self.lstm1_size, return_sequences=True, return_state=True))
        self.model.add(Attention())
        self.model.add(LSTM(self.lstm2_size, return_sequences=True, return_state=True))
        self.model.add(Dense(self.dense_size, activation='sigmoid'))

        self.optimizer(Adam(learning_rate=self.learning_rate))

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    for i in range(0, len(train_inputs), model.batch_size):
        batch_inputs = train_inputs[i:i+model.batch_size]
        batch_labels = train_labels[i:i+model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model(curr_inputs)
            loss = model.loss(predictions, curr_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("acc: ", model.accuracy(predictions, curr_labels))

def test(model, test_inputs, test_labels):
    total_accuracy = []
    for i in range(0, len(test_labels), model.batch_size):
        batch_inputs = test_inputs[i:i+model.batch_size]
        batch_labels = test_labels[i:i+(batch*model.window_size)]

        predictions = model.call(batch_inputs)
        total_accuracy.append(model.accuracy(predictions, batch_labels))

    return sum(total_accuracy) / len(total_accuracy)

def main():
    pos_fp = 'data/sentiment/P.txt'
    neg_fp = 'data/sentiment/N.txt'
    sarc_fp = 'data/sentiment/S.txt'

    print("Preprocessing data...")
    pos_vec, neg_vec, _, vocab, _ = get_data(pos_fp, neg_fp, sarc_fp)
    pos_labels = np.ones(shape(pos_vec)[0])
    neg_labels = np.zeros(shape(neg_vec)[0])

    inputs = np.concatenate((pos_vec, neg_vec))
    labels = np.concatenate((pos_labels, neg_labels))

    train_x, test_x, train_y, test_y = sk.train_test_split(inputs, labels, test_size=0.2, random_state=42)
    print("Preprocessing complete.\n")

    vocab_size = len(vocab)
    model = SentimentClassifier(vocab_size)

    print('Training sentiment classifier...')
    train(model, train_x, train_y)
    print("Training complete.\n")

    print('Testing sentiment classifier...')
    accuracy = test(model, test_x, test_y)
    print("Testing complete.\n")
    print("Final testing accuracy: ", accuracy)

if __name__ == '__main__':
	main()
