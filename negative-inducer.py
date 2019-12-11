import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM, Attention, concatenate

import numpy as np

import os
import argparse
from preprocess import get_data
from classifier import SentimentClassifier
from classifier import train as classifier_train
from classifier import test as classifier_test
import sklearn.model_selection as sk

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='sarc2seq')

parser.add_argument('--out-dir', type=str, default='negative-inducer/output',
                    help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--batch-size', type=int, default=64,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
                    help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=1,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--log-every', type=int, default=7,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()

## --------------------------------------------------------------------------------------

class NegativeInducer(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The model for the generator network is defined here. 
        """
        super(NegativeInducer, self).__init__()
        # TODO: Define the model, loss, and optimizer
        
        self.embedding_dim = 500
        self.hidden_size_rnn = 500

        self.batch_size = args.batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learn_rate)

        self.encoder_embedding = Embedding(vocab_size, embedding_dim)
        self.encoder = LSTM(self.hidden_size_rnn, return_sequences=True, return_state=True)

        self.decoder_embedding = Embedding(vocab_size, embedding_dim)
        self.decoder = LSTM(self.hidden_size_rnn, return_sequences=True, return_state=True)

        self.attention = Attention() #<-???????
        # original paper used bahdanau attention, but luong attention should be fine probably

        self.ff = TimeDistributed(Dense(self.vocab_size, activation="softmax"))
        # next : add copy mechanism to this :D


    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        Executes the generator model on the random noise vectors.

        :param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

        :return: prescaled generated images, shape=[batch_size, height, width, channel]
        """
        # TODO: Call the forward pass

        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)
        decoder_input = tf.convert_to_tensor(decoder_input, dtype=tf.int64)
    
        neut_embeddings = self.encoder_embeddings(encoder_input)
        #[batch_size x window_size x vocab_size]
        encoder_output, encoder_last, encoder_cell = self.encoder_rnn(inputs=french_embeddings)
        encoder_final_state = [encoder_last, encoder_cell]

        neg_embeddings = self.decoder_embeddings(decoder_input)
        decoder_out, _, _ = self.decoder_rnn(inputs=neg_embeddings, initial_state=encoder_final_state)

        context = self.attention([decoder_out, encoder_out]) # timedistributed somewhere???
        decoder_and_context = concatenate([decoder_out, context])
        out = self.ff(decoder_and_context)

        return out

    @tf.function
    def loss_function(self, prbs, labels, mask):
        """
        Outputs the loss given the discriminator output on the generated images.

        :param disc_fake_output: the discrimator output on the generated images, shape=[batch_size,1]

        :return: loss, the cross entropy loss, scalar
        """
        # TODO: Calculate the loss

        return tf.reduce_mean(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs),mask))

## --------------------------------------------------------------------------------------

# Train the model for one epoch.
def train(model, train_x, train_y, manager, pad_index):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return: None
    """
    encoder_input = train_x
    decoder_input = train_y[:, :-1]
    labels = train_y[:, 1:]

    num_examples = tf.shape(train_x)[0]
    for b in range(0, num_examples, model.batch_size):
        with tf.GradientTape() as tape:
            output = model(encoder_input[b:b+model.batch_size], decoder_input[b:b+model.batch_size])
            mask = (labels[b:b+model.batch_size] != pad_index)
            loss = model.loss_function(output, labels[b:b+model.batch_size], mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Save
        if iteration % args.save_every == 0:
            manager.save()

        if iteration % 500 == 0:
            print('**** LOSS: %g ****' % loss)

def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy

def test(model, test_x, test_y):
    """
    Test the model.

    :param model: negative inducer model

    :return: None
    """
    # TODO: do things somehow. bleu score? how did they do it in the seq2seq homework???
    
    encoder_input = train_x
    decoder_input = train_y[:, :-1]
    labels = train_y[:, 1:]

    num_windows = tf.shape(encoder_input)[0]
    total_loss = 0
    num_non_padded_correct = 0
    num_non_padded = 0
    for b in range(0, num_windows, model.batch_size):
        probs = model(encoder_input[b:b+model.batch_size], decoder_input[b:b+model.batch_size])
        mask = (labels[b:b+model.batch_size] != eng_padding_index)
        loss = model.loss_function(probs, labels[b:b+model.batch_size], mask)
        accuracy = model.accuracy_function(probs, labels[b:b+model.batch_size], mask)

        non_padded = tf.reduce_sum(tf.cast(mask, tf.float32))
        num_non_padded_correct += accuracy * non_padded
        total_loss += loss * non_padded
        num_non_padded += non_padded
    perplexity = tf.math.exp(total_loss / num_non_padded)
    per_symbol_accuracy = num_non_padded_correct / num_non_padded

    return perplexity,per_symbol_accuracy

## --------------------------------------------------------------------------------------

def main():
    neg_fp = 'data/sentiment/N.txt'
    neut_neg_fp = 'negative-inducer/NN.csv'

    vocab_size = 108706

    if not os.path.exists(neut_neg_fp):
        # literally just copied from classifier
        pos_fp = 'data/sentiment/P.txt'
        sarc_fp = 'data/sentiment/S.txt'


        print("Preprocessing data...")
        pos_vec, neg_vec, _, vocab, _ = get_data(pos_fp, neg_fp, sarc_fp)
        pos_labels = np.ones(pos_vec.shape[0])
        neg_labels = np.zeros(neg_vec.shape[0])

        inputs = np.concatenate((pos_vec, neg_vec))
        labels = np.concatenate((pos_labels, neg_labels))

        # save as csv file
        inputs = np.asarray(inputs)
        labels = np.asarray(labels)
        np.savetxt('inputs.csv', inputs, delimiter=',')
        np.savetxt('labels.csv', labels, delimiter=',')

        inputs = np.loadtxt('inputs.csv', delimiter=',')
        labels = np.loadtxt('labels.csv', delimiter=',')

        train_x, test_x, train_y, test_y = sk.train_test_split(inputs, labels, test_size=0.2, random_state=42)
        print("Preprocessing complete.\n")

        SCModel = SentimentClassifier(vocab_size)

        print('Training sentiment classifier...')
        classifier_train(SCModel, train_x, train_y)
        print("Training complete.\n")

        print('Testing sentiment classifier...')
        accuracy = classifier_test(model, test_x, test_y)
        print("Testing complete.\n")
        print("Final testing accuracy: ", accuracy)

        neutralized = SCModel(neg_vec)
        np.savetxt(neut_neg_fp, neutralized, delimiter=',')

    model = NegativeInducer(vocab_size)
    
    # For saving/loading models
    checkpoint_dir = 'negative-inducer/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint or args.mode == 'test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint) 

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                inputs = np.loadtxt(neut_neg_fp, delimiter=',')
                _, labels, _, vocab, pad_index = get_data(pos_fp, neg_fp, sarc_fp)
                train_x, test_x, train_y, test_y = sk.train_test_split(inputs, labels, test_size=0.2, random_state=42)
                for epoch in range(0, args.num_epochs):
                    train(model, train_x, train_y, manager, pad_index)
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
                perplexity,per_symbol_accuracy = test(model, test_x, test_y)
                print('**** test perplexity: %g, per_symbol_accuracy**** %g' % (perplexity, per_symbol_accuracy))
            if args.mode == 'test':
                test(model)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()


