import collections
import queue
import threading
import time

import numpy as np

from w2v import neuralnet, settings


class Vocabulary():
    def __init__(self, sample):
        self.total_token = 0
        self.seq_no_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.subsampling_rate = {}
        self.counter = collections.Counter()
        self.sample = sample

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
        self.total_token += len(tokens)
        self.counter.update(tokens)

    def build(self):
        # Counterを使って単語を出現頻度順に並べてある
        def z(count):
            return count / self.total_token
        for _, count in self.counter.items():
            if count <= settings.VOCABULARY_MIN_COUNT:
                break
        for tok, count in self.counter.most_common():
            if count <= settings.VOCABULARY_MIN_COUNT:
                break
            self.token_to_id[tok] = self.seq_no_id
            self.id_to_token[self.seq_no_id] = tok
            subsample_rate = (np.sqrt(z(count) / self.sample) + 1.0) \
                * self.sample / z(count)
            if subsample_rate >= 1:
                subsample_rate = 1.0
            self.subsampling_rate[tok] = subsample_rate
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
                if self.has(t) and self.subsampling:
                    one_hot[self.token_to_id[t]] += 1.0
            return [one_hot]

    def subsampling(self, token):
        p = self.subsampling_rate[token]
        return np.random.binomial(n=1, p=p)


class Word2Vec():
    # Skip Gram Model
    def __init__(self, window, alpha, size,
                 sample=0.001, codec='one_hot'):
        self.window_size = window
        self.nn = neuralnet.NeuralNetwork()
        self.alpha = alpha
        self.size = size
        self.codec = codec
        self.w = None
        self.w_ = None
        self.vocab = Vocabulary(sample)

    def _vectorize(self, data):
        if self.codec == 'one_hot':
            return self.vocab.one_hot(data)
        elif self.codec == 'huffman':
            return self.vocab.huffman(data)

    def make_model(self, data):
        for tokens in data:
            self.vocab.add(tokens)
        self.vocab.build()
        voca_dim = self.vocab.seq_no_id
        self.nn.prepare_model(self.alpha, self.size, voca_dim)
        self.nn.prepare_session()

    def _slice_window(self, tokens):
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
            for in_token, out_tokens in self._slice_window(tokens):
                if self.vocab.has(in_token) and self.vocab.has(out_tokens):
                    in_vecs.extend(self._vectorize(in_token))
                    out_vecs.extend(self._vectorize(out_tokens))
            q.put((np.array(in_vecs), np.array(out_vecs)))
        q.put((None, None))
        th.join()

        endtime = time.time()
        print(endtime - starttime)
        self.w = th.w
        self.w_ = th.w_
        return th.w, th.w_

    def _softmax(self, vec):
        vec_exp = np.exp(vec)
        vec_sum = np.sum(vec_exp)
        return vec_exp / vec_sum

    def most_similar(self, token, rank):
        input_onehot = self._vectorize(token)[0]
        wx = np.dot(input_onehot, self.w)
        predict = np.dot(wx, self.w_)
        predict /= np.linalg.norm(predict)
        pred_softmax = self._softmax(predict)
        print(np.sum(pred_softmax), np.max(pred_softmax))
        # softmax関数に通して0-1の値にした後ソートして上位を類似ワードとする
        sort = pred_softmax.argsort()[::-1]
        most_similar = []
        for i in sort[:rank]:
            most_similar.append((predict[i], self.vocab.id_to_token[i]))
        return most_similar

    def _cos_sim(self, vec1, vec2):
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
        cos_sim = np.dot(vec1, vec2.T)
        return cos_sim

    def most_similar_w_base(self, token, rank):
        code = self._vectorize(token)[0]
        wx = np.dot(code, self.w)
        most_similar = []
        # 重みベクトルWのコサイン類似度を測ったもの
        token_id = self.vocab.token_to_id[token]
        for i, vec in enumerate(self.w):
            if i == token_id:
                # 同じワードなら飛ばす
                continue
            wx /= np.linalg.norm(wx)
            vec /= np.linalg.norm(vec)
            cos_sim = np.dot(vec, wx.T)
            most_similar.append((cos_sim, self.vocab.id_to_token[i]))
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
