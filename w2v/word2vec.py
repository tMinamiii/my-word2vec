import collections
import queue
import threading

import time
import numpy as np
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
        # Counterを使って単語を出現頻度順に並べてある
        for tok, count in self.counter.most_common():
            if count >= settings.VOCABULARY_MAX_COUNT:
                print(tok)
                continue
            if count <= settings.VOCABULARY_MIN_COUNT:
                break
            self.token_to_id[tok] = self.seq_no_id
            self.id_to_token[self.seq_no_id] = tok
            self.seq_no_id += 1

    def huffman(self, data):
        if type(data) == str:
            dim = self.seq_no_id
            huffman = np.zeros(dim)
            # 入力層用
            if not self.has(data):
                return [huffman]
            seq_no = self.token_to_id[data]
            for i in range(seq_no + 1):
                huffman[i] = 1.0
            return [huffman]
        elif type(data) == list:
            # 出力層用
            dim = self.seq_no_id
            huffman = [0.0] * dim
            for t in data:
                if not self.has(t):
                    continue
                seq_no = self.token_to_id[t]
                for i in range(seq_no + 1):
                    huffman[i] += 1.0
            return [np.array(huffman)]

    def one_hot(self, data):
        if type(data) == str:
            # 入力層用
            dim = self.seq_no_id
            one_hot = np.zeros(dim)
            if not self.has(data):
                return [one_hot]
            one_hot[self.token_to_id[data]] = 1.0
            return [one_hot]
        elif type(data) == list:
            # 出力層用
            dim = self.seq_no_id
            one_hot = np.zeros(dim)
            for t in data:
                if self.has(t):
                    one_hot[self.token_to_id[t]] += 1.0
            # return [one_hot / np.linalg.norm(one_hot)]
            return [one_hot]


class Word2Vec():
    # Skip Gram Model
    def __init__(self, window, alpha, size, codec='one_hot'):
        self.window_size = window
        self.nn = neuralnet.NeuralNetwork()
        self.alpha = alpha
        self.size = size
        self.codec = codec
        self.w = None
        self.w_ = None
        self.vocab = None

    def vectorize(self, data):
        if self.codec == 'one_hot':
            return self.vocab.one_hot(data)
        elif self.codec == 'huffman':
            return self.vocab.huffman(data)

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
        q = queue.Queue(50)
        th = SessionRunner(nn, q)
        th.start()
        starttime = time.time()
        for tokens in data:
            in_vecs = []
            out_vecs = []
            for in_token, out_tokens in self.slice_window(tokens):
                if self.vocab.has(in_token) and self.vocab.has(out_tokens):
                    in_vecs.extend(self.vectorize(in_token))
                    out_vecs.extend(self.vectorize(out_tokens))
            q.put((np.array(in_vecs), np.array(out_vecs)))
        q.put((None, None))
        th.join()

        endtime = time.time()
        print(endtime - starttime)
        self.w = th.w
        self.w_ = th.w_
        return th.w, th.w_

    def most_similar(self, token, rank):
        code = self.vectorize(token)[0]
        wx = np.dot(code, self.w)
        predict = np.dot(wx, self.w_)
        if False:
            sort = predict.argsort()[::-1]
            most_similar = [self.vocab.id_to_token[i] for i in sort[:rank]]
            return most_similar
        else:
            most_similar = []
            for tok, seq_no in self.vocab.token_to_id.items():
                predict = predict / np.linalg.norm(predict)
                vec = self.vectorize(tok)[0]
                vec = vec / np.linalg.norm(vec)
                cos_sim = np.dot(vec, predict)
                most_similar.append((cos_sim, tok))
                most_similar = sorted(most_similar, reverse=True)
                most_similar = most_similar[:10]
            return np.array(most_similar)[:, :]


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
