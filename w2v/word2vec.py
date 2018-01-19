from collections import Counter
import tensorflow as tf
from w2v import utils
import numpy as np


def main():
    token_gen = utils.find_and_load_token_files()
    all_data = [line for line in token_gen]
    # data = ['I want to eat an apple everyday', 'I can fly', 'I love apple']
    w2v = Word2Vec(all_data, 5,  0.0001, 100)
    w2v.make_model()
    print('finish build')
    for _ in range(1):
        w2v.train()


# Word2Vec の学習に用いる各単語のonehotベクトルを作成する
class Vocabulary():
    token_to_id = {}
    seq_no_id = 0

    def build(self, tokens):
        token_counter = Counter(tokens)
        self.update_token_dics(token_counter)

    def update_token_dics(self, token_counter: Counter):
        for tok, _ in token_counter.items():
            if tok not in self.token_to_id:
                self.token_to_id[tok] = self.seq_no_id
                self.seq_no_id += 1

    def one_hot(self, token):
        dim = self.seq_no_id + 1
        if type(token) == str:
            one_hot = np.zeros(dim)
            one_hot[self.token_to_id[token]] = 1.0
            return [one_hot]
        if type(token) == list:
            one_hot = np.zeros(dim)
            for t in token:
                one_hot[self.token_to_id[t]] += 1.0
            return [one_hot]


class Word2Vec():
    # Skip Gram Model
    def __init__(self, data, window_size, learning_rate, dim):
        self.data = data
        self._vocabulary = Vocabulary()
        self.window_size = window_size
        self._nn = NeuralNetwork()
        self.learning_rate = learning_rate
        self.dim = dim

    def make_model(self):
        for line in self.data:
            t = line.split(' ')
            self._vocabulary.build(t)
        voca_dim = self._vocabulary.seq_no_id + 1
        self._nn.prepare_model(self.learning_rate, self.dim, voca_dim)
        self._nn.prepare_session()

    def slice_window(self, tokens):
        position = 0
        for position in range(len(tokens)):
            window_begin = position - self.window_size
            window_end = position + self.window_size
            if window_begin < 0:
                window_begin = 0
            window = tokens[window_begin:window_end]
            input_token = tokens[position]
            output_token = window[:self.window_size] + \
                window[self.window_size + 1:]
            yield input_token, output_token

    def train(self):
        i = 0
        w_out = None
        for line in self.data:
            i += 1
            print('{}/{}'.format(i, len(self.data)))
            tokens = line.split(' ')
            input_vecs = []
            output_vecs = []
            window_gen = self.slice_window(tokens)
            for window in window_gen:
                input_token, output_tokens = window
                in_vec = self._vocabulary.one_hot(input_token)
                out_vec = self._vocabulary.one_hot(output_tokens)
                input_vecs += in_vec
                output_vecs += out_vec
            w_out, _ = self._nn.sess.run([self._nn.w1,
                                          self._nn.train_step],
                                         feed_dict={
                self._nn.x: input_vecs,
                self._nn.t: output_vecs,
            })
        return w_out


class NeuralNetwork:
    def prepare_model(self, learning_rate, num_units, vec_dim):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, vec_dim])

        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
            b1 = tf.Variable(tf.zeros(num_units))
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, vec_dim]))
            b0 = tf.Variable(tf.zeros([vec_dim]))
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, vec_dim])
            loss = -1 * tf.reduce_sum(t * tf.log(p))
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)

        # with tf.name_scope('evaluator'):
        # correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tf.summary.scalar('loss', loss)
        # tf.summary.scalar('accuracy', accuracy)

        self.w1 = w1
        self.b1 = b1
        self.x = x
        self.t = t
        self.p = p
        self.train_step = train_step
        self.loss = loss
        # self.accuracy = accuracy

    def prepare_session(self):
        # sess = tf.InteractiveSession()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        # saver = tf.train.Saver()
        # summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(logfile, sess.graph)
        # self.saver = saver
        # self.summary = summary
        # self.writer = writer
