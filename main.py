import random

from w2v import settings, utils, word2vec


def main():
    token_gen = utils.find_and_load_token_files()
    token_gen = [line for _, line in token_gen]
    # print(sorted(token_gen)[0])
    token_gen = sorted(token_gen)[:settings.NUM_DOCUMENTS]
    # token_gen= [('aa','I want to eat an apple everyday'.split(' '))]
    w2v = word2vec.Word2Vec(window=settings.WINDOW,
                            alpha=settings.ALPHA,
                            size=settings.SIZE,
                            sample=settings.SAMPLE,
                            codec='one_hot')
    w2v.make_model(token_gen)
    word = 'iphone'
    rank = 10
    print(w2v.vocab.seq_no_id)
    print('finish build')
    for i in range(100):
        # token_gen = utils.find_and_load_token_files()
        random.shuffle(token_gen)
        w2v.train(token_gen)
        if i == 0 or (i + 1) % 5 == 0:
            print("epoch = {} -----------".format(i + 1))
            most_similar = w2v.most_similar(word, rank)
            print_similar(word, most_similar)
            # most_similar = w2v.most_similar_w(word, rank)
            # print_similar(word, most_similar)


def print_similar(word, most_similar):
    print('{} ====='.format(word))
    for i, (num, token) in enumerate(most_similar):
        print('{} : {} \t {}'.format(i, token, num))


if __name__ == '__main__':
    main()
