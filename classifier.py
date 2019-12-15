import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Attention, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
from preprocess import get_data
import math
import os
import sklearn.model_selection as sk
from attention import SeqSelfAttention


class SentimentClassifier(Model):
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
        self.batch_size = 100

        self.lstm1_size = 200
        self.lstm2_size = 150
        self.dense_size = 2
        self.learning_rate = 0.01

        self.E1 = Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.LSTM1 = LSTM(self.lstm1_size, return_sequences=True, return_state=True)
        # self.A1 = Attention()
        self.A1 = SeqSelfAttention(attention_activation='sigmoid', return_attention=True)
        self.LSTM2 = LSTM(self.lstm2_size, return_sequences=True, return_state=True)
        self.pool = GlobalAveragePooling1D()
        self.D1 = Dense(self.dense_size, activation='sigmoid')

        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.is_trained = False

    def call(self, inputs, neutralizing=False):
        embeddings = self.E1(inputs)
        lstm1_out, _, _ = self.LSTM1(embeddings)
        # attention_out = self.A1([lstm1_out, lstm1_out])
        attention_out, att_weights = self.A1(lstm1_out)
        if neutralizing:
            return att_weights
        lstm2_out, _, _ = self.LSTM2(attention_out)
        pool_out = self.pool(lstm2_out)
        dense_out = self.D1(pool_out)
        return dense_out

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    for i in range(0, len(train_inputs), model.batch_size):
        batch_inputs = train_inputs[i:i+model.batch_size]
        batch_labels = train_labels[i:i+model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = model.loss(predictions, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 500 == 0:
            print("acc: ", model.accuracy(predictions, batch_labels))

    model.is_trained = True

def test(model, test_inputs, test_labels):
    total_accuracy = []
    for i in range(0, len(test_labels), model.batch_size):
        batch_inputs = test_inputs[i:i+model.batch_size]
        batch_labels = test_labels[i:i+model.batch_size]

        predictions = model.call(batch_inputs)
        total_accuracy.append(model.accuracy(predictions, batch_labels))

    return sum(total_accuracy) / len(total_accuracy)

def main():
    pos_fp = 'data/sentiment/P.txt'
    neg_fp = 'data/sentiment/N.txt'
    sarc_fp = 'data/sentiment/S.txt'
    classifier_inputs_fp = 'classifier_inputs.csv'
    classifier_labels_fp = 'classifier_labels.csv'

    print("Preprocessing data...")

    # if preprocessing has already been done and stored, read from file
    # if os.path.exists(classifier_inputs_fp) and os.path.exists(classifier_labels_fp):
    if False:
        inputs = np.loadtxt(classifier_inputs_fp, delimiter=',')
        labels = np.loadtxt(classifier_labels_fp, delimiter=',')
    else: # perform preprocessing and write to file
        pos_vec, neg_vec, _, vocab, _ = get_data(pos_fp, neg_fp, sarc_fp)
        exit()
        pos_labels = np.ones(pos_vec.shape[0])
        neg_labels = np.zeros(neg_vec.shape[0])

        inputs = np.concatenate((pos_vec, neg_vec))
        labels = np.concatenate((pos_labels, neg_labels))
        inputs = np.asarray(inputs)
        labels = np.asarray(labels)
        np.savetxt(classifier_inputs_fp, inputs, delimiter=',')
        np.savetxt(classifier_labels_fp, labels, delimiter=',')

    train_x, test_x, train_y, test_y = sk.train_test_split(inputs, labels, test_size=0.2, random_state=42)
    print("Preprocessing complete.\n")

    model = SentimentClassifier(len(vocab))

    print('Training sentiment classifier...')
    train(model, train_x, train_y)
    print("Training complete.\n")

    print('Testing sentiment classifier...')
    accuracy = test(model, test_x, test_y)
    print("Testing complete.\n")
    print("Final testing accuracy: ", accuracy)

if __name__ == '__main__':
    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)

    device = 'GPU:0' if gpu_available else 'CPU:0'
    try:
        with tf.device('/device:' + device):
    	       main()
    except RuntimeError as e:
        print(e)
