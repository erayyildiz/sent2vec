import random
import re

import keras.backend as K
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.layers import Embedding, Input, Dot, Softmax, Permute, Layer, Dropout, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.regularizers import l1
from keras.initializers import glorot_normal
from tqdm import tqdm


class MeanPool(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:

            if mask.dtype != K.floatx():
                mask = K.cast(mask, K.floatx())
            if len(mask.shape) != 2:
                raise ValueError(
                    'mask should have `shape=(samples, time)`, '
                    'got {}'.format(mask.shape))

            # mask (batch, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, x_dim, time)
            mask = tf.transpose(mask, [0, 2, 1])
            # mask (batch, time, x_dim)
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]


class Sent2Vec:
    sentence_splitter_regex = re.compile(r'(([\.\?!\:]+)|:[\)\(])')
    splitter_regex = re.compile(r'[\t\r\n ]+')
    punc_remove_regex = re.compile(r'[\.,\?!\"\(\)\[\]\+\%\&\/\\=\*\-\|_]')

    def __init__(self, input_txt_file, batch_size=32, lower=True, asciify=False, embedding_dim=100, negative_samples=10,
                 num_epoch=5, min_token_count=5, min_ngram_count=5, stop_words_file=None, max_sentences=0, max_len=20):

        self.ngram_vocab = dict()
        self.output_vocab = dict()
        self.max_len = 0
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        self.min_token_count = min_token_count
        self.min_ngram_count = min_ngram_count
        self.lower = lower
        self.asciify = asciify
        self.stop_words = set()
        self.max_len = max_len
        if stop_words_file:
            print('Reading stop words')
            self.read_stop_words(stop_words_file)

        print('Reading sentences')
        self.sentences = []
        with open(input_txt_file, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(tqdm(f)):
                for sentence in self.split_sentences(line.strip()):
                    tokens = self.tokenize(sentence)
                    if self.max_len >= len(tokens) > 3:
                        self.sentences.append(tokens)
                if 0 < max_sentences <= len(self.sentences):
                    break

        self.total_token_count = 0
        self.keep_probs = {}
        self.build_vocab()
        self.output_tokens_set = set(list(self.output_vocab.keys()))
        self.negative_samples = min(self.negative_samples, len(self.output_tokens_set) - 1)
        self.model = None

        print('Counting training samples')
        self.num_batches = self.get_data_size()
        print('Num batches in training data: {}'.format(self.num_batches))

        # print('Constructing dataset')
        # self.X1, self.X2, self.Y = self.construct_dataset()
        # print('Done.')

    def get_data(self):
        X1 = []
        X2 = []
        Y = []
        random.shuffle(self.sentences)
        while True:
            for sentence in self.sentences:
                for _x1, _x2 in self.extract_instances(sentence):
                    X1.append(_x1)
                    X2.append(_x2)
                    Y.append([1] + [0] * self.negative_samples)
                    if len(X1) == self.batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(Y)
                        X1 = []
                        X2 = []
                        Y = []

    def get_data_size(self):
        res = 0
        X1 = []

        for sentence in tqdm(self.sentences):
            for _x1, _x2 in self.extract_instances(sentence):
                X1.append(_x1)
                if len(X1) == self.batch_size:
                    res += 1
                    X1 = []
        return res

    def construct_dataset(self):
        X1 = []
        X2 = []
        Y = []
        for sentence in tqdm(self.sentences):
            for _x1, _x2 in self.extract_instances(sentence):
                X1.append(_x1)
                X2.append(_x2)
                Y.append([1] + [0] * self.negative_samples)
        return np.array(X1), np.array(X2), np.array(Y)

    def train(self, num_epoch=5):
        embedding_dict = {}
        self.build_model()
        self.construct_dataset()
        for epoch in range(num_epoch):
            self.model.fit_generator(self.get_data(), steps_per_epoch=self.num_batches, verbose=1)
        # self.model.fit([self.X1, self.X2], self.Y, batch_size=self.batch_size, epochs=num_epoch)
        embedding_layer = self.model.get_layer('embeddings')
        weights = embedding_layer.get_weights()[0]
        for ngram, v in self.ngram_vocab.items():
            embedding_dict[ngram] = weights[v]

        with open('sent2vec.embeddings', 'w', encoding='UTF-8') as f:
            f.write('{} {}\n'.format(len(embedding_dict), self.embedding_dim))
            for k, v in embedding_dict.items():
                f.write('{} {}\n'.format(k, ' '.join([str(x) for x in v.tolist()])))

        return KeyedVectors.load_word2vec_format('sent2vec.embeddings')

    def read_stop_words(self, stop_words_file):
        stop_words = []
        with open(stop_words_file, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                if self.asciify:
                    stop_words.append(self.to_ascii(self.to_lower(line.strip())))
                else:
                    stop_words.append(self.to_lower(line.strip()))
        self.stop_words = set(stop_words)

    def build_vocab(self):
        print('Counting word and ngram occurences')
        ngram_counts = {}
        token_counts = {}
        for sentence in tqdm(self.sentences):
            for i, token in enumerate(sentence):
                if token in self.stop_words:
                    continue
                self.total_token_count += 1
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1

                right_tokens = sentence[:i]
                left_tokens = sentence[i + 1:]
                ngrams = self.extract_ngrams(right_tokens) + self.extract_ngrams(left_tokens)

                for ngram in ngrams:
                    if ngram in ngram_counts:
                        ngram_counts[ngram] += 1
                    else:
                        ngram_counts[ngram] = 1

                if len(ngrams) > self.max_len:
                    self.max_len = len(ngrams)

        print('Building token output vocab')

        for token, count in token_counts.items():
            if not token:
                continue
            if count >= self.min_token_count:
                self.output_vocab[token] = len(self.output_vocab)
                z = count * 1.0 / self.total_token_count
                self.keep_probs[token] = (np.sqrt(z / 0.0001) + 1) * (0.0001 / z)

        print('Building ngram input vocab')
        for ngram, count in ngram_counts.items():
            if not ngram:
                continue
            if count >= self.min_ngram_count:
                self.ngram_vocab[ngram] = len(self.ngram_vocab) + 1

        print('Finished. Num input ngrams: {}, Num target tokens: {}'.format(len(self.ngram_vocab),
                                                                             len(self.output_vocab)))

    def build_model(self):
        print('Building Model')

        input_embedding_table = Embedding(len(self.ngram_vocab) + 1, self.embedding_dim,
                                          # embeddings_initializer=glorot_normal(seed=None),
                                          # mask_zero=True,
                                          name='embeddings',
                                          # embeddings_regularizer=l1()
                                          )
        output_embedding_table = Embedding(len(self.output_vocab) + 1, self.embedding_dim,
                                           # embeddings_initializer=glorot_normal(seed=None),
                                           # embeddings_regularizer=l1()
                                           )
        sentence_inputs = Input(shape=(self.max_len,))
        target_inputs = Input(shape=(self.negative_samples + 1,))
        sentence_embeddings = input_embedding_table(sentence_inputs)
        # sentence_embeddings = Dropout(0.3)(sentence_embeddings)
        target_embeddings = output_embedding_table(target_inputs)
        # target_embeddings = Dropout(0.3)(target_embeddings)
        target_embeddings = Permute((2, 1))(target_embeddings)
        # sentence_embedding = MeanPool()(sentence_embeddings)
        sentence_embedding = GlobalAveragePooling1D()(sentence_embeddings)
        scores = Dot(axes=1)([target_embeddings, sentence_embedding])
        probs = Softmax()(scores)
        self.model = Model(inputs=[sentence_inputs, target_inputs], outputs=[probs], name='sent2vec')
        self.model.summary()
        # sgd = SGD(lr=0.001, momentum=0.0, decay=0.1, nesterov=False)
        self.model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='Adam')

    @classmethod
    def split_sentences(cls, txt):
        return cls.sentence_splitter_regex.split(txt)

    @classmethod
    def to_lower(cls, txt):
        res = txt.replace('İ', 'i')
        res = res.replace('Ü', 'ü')
        res = res.replace('Ğ', 'ğ')
        res = res.replace('Ö', 'ö')
        res = res.replace('Ş', 'ş')
        res = res.replace('Ç', 'ç')
        res = res.replace('I', 'ı')
        return res.lower()

    @classmethod
    def to_ascii(cls, txt):
        res = txt.replace('ı', 'i')
        res = res.replace('ü', 'u')
        res = res.replace('ğ', 'g')
        res = res.replace('ö', 'o')
        res = res.replace('ş', 's')
        res = res.replace('ç', 'c')
        return res

    def encode_ngrams(self, ngrams):
        res = np.zeros((self.max_len,))

        ix = 0
        for ngram in ngrams:
            if ngram in self.ngram_vocab:
                res[ix] = self.ngram_vocab[ngram]
                ix += 1

        return res

    def encode_token(self, token, ngrams):
        res = np.zeros((self.negative_samples + 1,))
        res[0] = self.output_vocab[token]

        # TODO SUB SAMPLING
        candidate_negative_tokens = set(random.sample(self.output_tokens_set, self.negative_samples))
        if token in candidate_negative_tokens:
            candidate_negative_tokens = candidate_negative_tokens - {token}
            candidate_negative_tokens = candidate_negative_tokens - set(ngrams)
        negative_tokens = list(candidate_negative_tokens)

        for i, negative_token in enumerate(negative_tokens):
            if i >= self.negative_samples:
                break
            res[i + 1] = self.output_vocab[negative_token]

        return res

    def extract_instances(self, tokens):
        for i, token in enumerate(tokens):
            if token not in self.output_vocab:
                continue

            if random.random() > self.keep_probs[token]:
                # print('Skipping token: {} - prob: {}'.format(token, self.keep_probs[token]))
                continue

            right_tokens = tokens[:i]
            left_tokens = tokens[i + 1:]
            ngrams = self.extract_ngrams(right_tokens) + self.extract_ngrams(left_tokens)
            # print('{} >> {}'.format(token, ngrams))
            ngrams_encoded = self.encode_ngrams(ngrams)
            # print('Ngrmas encoded: {}'.format(ngrams_encoded))
            target_tokens_encoded = self.encode_token(token, ngrams)
            # print('Target tokens encoded: {}'.format(target_tokens_encoded))
            yield ngrams_encoded, target_tokens_encoded

    def tokenize(self, txt):
        if self.lower:
            txt = self.to_lower(txt)
        if self.asciify:
            txt = self.to_ascii(txt)

        txt = self.punc_remove_regex.sub(' ', txt)
        tokens = self.splitter_regex.split(txt)
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def extract_ngrams(self, tokens, Ns=(2,)):
        res = []

        l = len(tokens)
        res += tokens

        for n in Ns[1:]:
            if l - n - 4 > 0:
                res += random.sample(['_'.join(tokens[start_ix: start_ix + n]) for start_ix in range(0, l - n + 1)],
                                     l - n - 4)

        return res


if __name__ == '__main__':
    sent2vec = Sent2Vec('trip_advisor.txt', stop_words_file='stopwords.txt', batch_size=128,
                        min_ngram_count=20, min_token_count=40, max_sentences=50000)
    embeddings = sent2vec.train(num_epoch=5)

