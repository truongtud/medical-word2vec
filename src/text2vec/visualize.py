import logging

import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.cleaner import Cleaner
from .embedding_vectorizer import Word2VecMeanEmbeddingVectorized, Doc2VecEmbeddingVectorized, \
    FastTextMeanEmbeddingVectorized
from .model_reader import ModelReader


class Visualize(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_reader = ModelReader()
        self.cleaner = Cleaner()

    def plot_word2_vec(self, words, result, color):
        plt.figure(figsize=(5, 5))
        plt.scatter(result[:, 0], result[:, 1], edgecolors='k', c=color)
        for word, (x, y) in zip(words, result):
            plt.text(x, y, word)
        plt.show()

    def plot_doc2_vec(self, docs, result, color):
        plt.figure(figsize=(5, 5))
        plt.scatter(result[:, 0], result[:, 1], edgecolors='k', c=color)
        for doc, (x, y) in zip(docs, result):
            words = word_tokenize(doc.lower(), language='german')
            plt.text(x, y, words)
        plt.show()

    def pca_word2vec(self, words, model_file):
        model = self.model_reader.get_doc2vec_model(model_file)
        words_vec = np.vstack([model[w] for w in words])
        pca = PCA(n_components=2)
        result = pca.fit_transform(words_vec)
        self.plot_word2_vec(words, result, 'blue')

    def pca_doc2vec(self, docs, model_file):
        ev = Doc2VecEmbeddingVectorized(model_file, 'doc2vec')
        docs_vec = np.vstack([ev.embedding(doc) for doc in docs])
        pca = PCA(n_components=2)
        result = pca.fit_transform(docs_vec)
        self.plot_doc2_vec(docs, result, color='red')

    def t_sne_doc2vec(self, docs, model_file):
        ev = Doc2VecEmbeddingVectorized(model_file, 'doc2vec')
        docs_vec = np.vstack([ev.embedding(doc) for doc in docs])
        result = TSNE(n_components=2).fit_transform(docs_vec)
        self.plot_doc2_vec(docs, result, color='red')

    def t_sne_word2vec(self, words, model_file):
        model = self.model_reader.get_word2vec_model(model_file)
        words_vec = np.vstack([model[w] for w in words])
        result = TSNE(n_components=2).fit_transform(words_vec)
        self.plot_word2_vec(words, result, 'blue')

    def most_similar_doc2vec(self, doc, model_file):
        ev = Doc2VecEmbeddingVectorized(model_file, 'doc2vec')
        infer_vector = ev.embedding(doc)
        si_words = ev.model.docvecs.most_similar([infer_vector], topn=10)
        print(si_words)

    def most_similar_word2vec(self, word, model_file):
        word = word.strip().lower()
        word = self.cleaner.replace_umlauts(word)
        model = self.model_reader.get_word2vec_model(model_file)
        vocabs = model.wv.vocab
        print(vocabs[word])
        si_words = model.wv.similar_by_word(word, topn=100)
        print(si_words)

    def most_similar_fasttext(self, word, model_file):
        word = word.strip().lower()
        word = self.cleaner.replace_umlauts(word)
        model = self.model_reader.get_fasttext_model(model_file)
        vocabs = model.wv.vocab
        print(vocabs[word])
        si_words = model.wv.similar_by_word(word, topn=100)
        print(si_words)

    def pca_mean_word2vec_texts(self, texts, model_file):
        ev = Word2VecMeanEmbeddingVectorized(model_file, 'word2vec')
        words_vec = np.vstack([ev.embedding(self.cleaner.clean(text)) for text in texts])
        pca = PCA(n_components=2)
        result = pca.fit_transform(words_vec)
        self.plot_word2_vec(texts, result, 'blue')

    def pca_mean_fasttext_texts(self, texts, model_file):
        ev = FastTextMeanEmbeddingVectorized(model_file, 'fasttext')
        words_vec = np.vstack([ev.embedding(self.cleaner.clean(text)) for text in texts])
        pca = PCA(n_components=2)
        result = pca.fit_transform(words_vec)
        self.plot_word2_vec(texts, result, 'blue')

    def compoundWord(self, compoundWord, model_file):
        compoundWord = compoundWord.strip().lower()
        compoundWord = self.cleaner.replace_umlauts(compoundWord)
        ev = Word2VecMeanEmbeddingVectorized(model_file, 'word2vec')
        vocabs = ev.model.wv.vocab
        ev.compound_word_vector(compoundWord, vocabs)
