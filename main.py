import random

import numpy as np

from w2v import settings, utils, word2vec


def main():
    token_gen = utils.find_and_load_token_files()
    token_gen = [line for _, line in token_gen]
    # print(sorted(token_gen)[0])
    token_gen = sorted(token_gen)[:settings.NUM_DOCUMENTS]
    # token_gen= [('aa','I want to eat an apple everyday'.split(' '))]

    w2v = word2vec.Word2Vec(window=15,  alpha=0.0005, size=200)
    w2v.make_model(token_gen)
    word = 'iphone'
    rank = 10
    print(w2v.vocab.seq_no_id)
    onehot = w2v.vocab.one_hot(word)[0]
    print('finish build')
    for i in range(100):
        # token_gen = utils.find_and_load_token_files()
        random.shuffle(token_gen)
        w, w_ = w2v.train(token_gen)
        r = np.dot(onehot, w)
        predict = np.dot(r, w_)
        sort = predict.argsort()[::-1]
        most_similar = [w2v.vocab.id_to_token[i] for i in sort[:rank]]
        if i == 0 or (i + 1) % 5 == 0:
            print("epoch = {} -----------".format(i + 1))
            print('{} =====> {}'.format(word, most_similar))


if __name__ == '__main__':
    main()
