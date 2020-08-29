from abc import abstractmethod

import numpy as np
from nltk.tokenize import word_tokenize

from utils.cleaner import Cleaner
from .model_reader import ModelReader

cleaner = Cleaner()


class EmbeddingVectorized(object):
    def __init__(self, model_path, model_type):
        self.model_reader = ModelReader()
        self.model = self.model_reader.get_model(model_path, model_type)
        self.dim = self.model.vector_size
        self.model_type = model_type

    @abstractmethod
    def embedding(self, text):
        pass

    # class Word2Vec(object):
    #   def _get_word2vec(self, word):
    #      return self.model[word] */

    def get_embedding_matrix(self):
        vocabs_size = len(self.model.wv.vocab)
        embedding_matrix = np.zeros((vocabs_size, self.dim))
        for i in range(vocabs_size):
            embedding_vector = self.model.wv[self.model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    """
    handle compound words that are not available in word2vec model. The best possible vector will be obtained using mean vector.
    For example, "Fußprellung"
    """

    def compound_word_vector(self, compound_word, vocabs):
        from CharSplit.char_split import split_compound
        possible_split_words = split_compound(compound_word)
        best_mean_vector = [np.zeros(self.dim)]
        best_split = []
        if len(possible_split_words) > 0:
            for result in possible_split_words:
                split_words = result[1:]
                if self.is_compound_word_in_vocabs(split_words, vocabs):
                    best_mean_vector = np.mean([self.model[word.lower()] for word in split_words if
                                                (not cleaner.is_stop_word(word))] or [np.zeros(self.dim)],
                                               axis=0)
                    best_split = split_words
                    break
        print('{0} {1} {2}'.format(compound_word, best_split, best_mean_vector))
        return best_mean_vector

    def is_compound_word_in_vocabs(self, split_Words, vocabs):
        all_split_words_in_vocabs = False
        for word in split_Words:
            if not (word.lower() in vocabs):
                return False
            else:
                all_split_words_in_vocabs = True
        return all_split_words_in_vocabs


class Word2VecMeanEmbeddingVectorized(EmbeddingVectorized):
    def embedding(self, text):
        vocabs = self.model.wv.vocab.keys()
        words = word_tokenize(text.lower(), language='german')
        # print(words)
        vectors = []
        for word in words:
            if word in vocabs and not cleaner.is_stop_word(word):
                vectors.append(self.model[word])
            elif not (word in vocabs) and not cleaner.is_stop_word(word):
                vectors.append(self.compound_word_vector(word, vocabs))
            else:
                vectors.append([np.zeros(self.dim)])
        # mean_vector=np.array(np.mean(vectors,axis=0))
        return np.array(np.mean(vectors, axis=0))

    def _infer_word2vec(self, word, model):
        v = model[word]
        return v


class FastTextMeanEmbeddingVectorized(EmbeddingVectorized):
    def embedding(self, text):
        vocabs = self.model.wv.vocab.keys()
        words = word_tokenize(text.lower(), language='german')
        # print(words)
        vectors = []
        for word in words:
            if word in vocabs and not cleaner.is_stop_word(word):
                vectors.append(self.model[word])
            elif not (word in vocabs) and not cleaner.is_stop_word(word):
                vectors.append(self.compound_word_vector(word, vocabs))
            else:
                vectors.append([np.zeros(self.dim)])
        # mean_vector=np.array(np.mean(vectors,axis=0))
        return np.array(np.mean(vectors, axis=0))

    def _infer_word2vec(self, word, model):
        v = model[word]
        return v


class Doc2VecEmbeddingVectorized(EmbeddingVectorized):
    def embedding(self, text):
        return self._infer_doc2vec(text, self.model)

    def _infer_doc2vec(self, text, model):
        words = word_tokenize(text.lower(), language='german')
        v = model.infer_vector(words)
        # print(v)
        return v


class EmbeddingVectorizedFactory(object):
    @staticmethod
    def create_embedding_vectorized(embedding_file, embedding_type):
        if embedding_type == 'word2vec':
            return Word2VecMeanEmbeddingVectorized(embedding_file, 'word2vec')
        elif embedding_type == 'doc2vec':
            return Doc2VecEmbeddingVectorized(embedding_file, 'doc2vec')
        elif embedding_type == 'fasttext':
            return FastTextMeanEmbeddingVectorized(embedding_file, 'fasttext')
