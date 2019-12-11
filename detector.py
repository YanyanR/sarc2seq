"""
This is a sarcasm detector that takes in a senentence,
and outputs a confidence score (the probability of being sarcastic).
"""
import tensorflow as tf
import numpy as np
from preprocess import get_data
import math
import os

class Model(tf.keras.Model):
    """
    architecture: similar to the sentiment classifier
    embedding + LSTM + attention + LSTM + Dense
    """
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 40
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
        att_output = self.att(lstm1_output)
        lstm2_output, lstm2_state1, lstm2_state2 = self.lstm2(att_output)
        final_output = self.D(lstm2_output)

        #TODO: where to put softmax? how to return the possibility?
        return final_output

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    for i in range(0, len(train_inputs), model.batch_size*model.window_size):
        curr_inputs = np.array(train_inputs[i:i+(model.batch_size*model.window_size)])
        batch = int(curr_inputs.shape[0] / model.window_size)
        curr_inputs = curr_inputs[:batch*model.window_size]
        curr_labels = train_labels[i:i+(batch*model.window_size)]

        curr_inputs = np.reshape(curr_inputs, [batch, model.window_size])

        # backprop
        with tf.GradientTape() as tape:
            predictions, _ = model.call(curr_inputs, None)
            loss = model.loss(predictions, curr_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    total_accuracy = []
    for i in range(0, len(test_labels), model.batch_size*model.window_size):
        curr_inputs = np.array(test_inputs[i:i+(model.batch_size*model.window_size)])
        batch = int(curr_inputs.shape[0] / model.window_size)
        curr_inputs = curr_inputs[:batch*model.window_size]
        curr_labels = test_labels[i:i+(batch*model.window_size)]

        curr_inputs = np.reshape(curr_inputs, [batch, model.window_size])

        predictions, _ = model.call(curr_inputs)
        total_accuracy.append(model.accuracy(predictions, curr_labels))

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
    train_x, train_y, test_x, test_y, vocab_size = get_data('./data/S.txt', './data/P.txt', './data/N.txt')
    print("finished data processing")
    # initialize model and tensorflow variables
    model = Model(vocab_size)

    # Set-up the training step
    train(model, train_x, train_y)
    # Set up the testing steps
    accuracy = test(model, test_x, test_y)
    print("accuracy: ", accuracy)

if __name__ == '__main__':
	main()