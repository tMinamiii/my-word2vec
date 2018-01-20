import collections
import queue
import threading

from numpy import zeros

from w2v import neuralnet, settings


class Vocabulary():
    def __init__(self):
        self.seq_no_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.counter = collections.Counter()

    def has(self, data):
        # 単語 or 単語リストが辞書にあるか無いかを判定する
        if type(data) == str:
            return data in self.token_to_id
        if type(data) == list:
            for t in data:
                if t in self.token_to_id:
                    return True
            return False

    def add(self, tokens):
        self.counter.update(tokens)

    def build(self):
        for tok, count in self.counter.most_common():
            if count <= settings.VOCABULARY_MIN_COUNT:
                break
            if tok not in self.token_to_id:
                self.token_to_id[tok] = self.seq_no_id
                self.id_to_token[self.seq_no_id] = tok
                self.seq_no_id += 1

    def one_hot(self, data):
        dim = self.seq_no_id
        if type(data) == str:
            # 入力層用
            one_hot = zeros(dim)
            one_hot[self.token_to_id[data]] = 1.0
            # print('input  {}'.format(one_hot))
            return [one_hot]
        if type(data) == list:
            # 出力層用
            one_hot = zeros(dim)
            for t in data:
                if self.has(t):
                    one_hot[self.token_to_id[t]] += 1.0
            # print('output {}'.format(one_hot))
            return [one_hot]


class Word2Vec():
    # Skip Gram Model
    def __init__(self, window, alpha, size):
        self.window_size = window
        self.nn = neuralnet.NeuralNetwork()
        self.alpha = alpha
        self.size = size
        self.w = None
        self.w_ = None
        self.vocab = None

    def make_model(self, data):
        self.vocab = Vocabulary()
        for tokens in data:
            self.vocab.add(tokens)
        self.vocab.build()
        voca_dim = self.vocab.seq_no_id
        self.nn.prepare_model(self.alpha, self.size, voca_dim)
        self.nn.prepare_session()

    def slice_window(self, tokens):
        ws = self.window_size
        for pos in range(len(tokens)):
            w_begin = pos - ws
            w_end = pos + ws
            if w_begin < 0:
                if pos == 0:
                    w_begin = 1
                else:
                    w_begin = 0
            input_token = tokens[pos]
            output_token = tokens[w_begin:pos] + tokens[pos + 1:w_end]
            yield input_token, output_token

    def train(self, data):
        nn = self.nn
        q = queue.Queue(2)
        th = SessionRunner(nn, q)
        th.start()
        in_vecs = []
        out_vecs = []
        for tokens in data:
            for in_token, out_tokens in self.slice_window(tokens):
                if self.vocab.has(in_token) and self.vocab.has(out_tokens):
                    in_vecs.extend(self.vocab.one_hot(in_token))
                    out_vecs.extend(self.vocab.one_hot(out_tokens))
            # w, w_ = self.run_session(input_vecs, output_vecs)
            q.put((in_vecs, out_vecs))
            in_vecs = []
            out_vecs = []
        q.put((None, None))
        th.join()

        # return w, w_
        return th.w, th.w_


class SessionRunner(threading.Thread):
    def __init__(self, nn, q):
        super(SessionRunner, self).__init__()
        self.nn = nn
        self.q = q

    def run(self):
        while True:
            input_vecs, output_vecs = self.q.get()
            if input_vecs is None and output_vecs is None:
                break
            w, w_, _ = self.nn.sess.run([self.nn.w1, self.nn.w0,
                                         self.nn.train_step],
                                        feed_dict={
                self.nn.x: input_vecs,
                self.nn.t: output_vecs,
            })
        self.w = w
        self.w_ = w_
