import logging
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec

from .raw_data_reader import RawDataReader

abspath = os.path.dirname(os.path.abspath(__file__))


class TrainerText2Vec:
    def __init__(self):
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    """
    Train doc2vec model
    """

    def _doc2vec(self, medical_texts):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(medical_texts)]
        model = Doc2Vec(documents, vector_size=100, window=2, workers=10, min_count=1)
        model.train(documents, total_examples=len(documents), epochs=10)
        model.save(os.path.join(abspath, "../vectors/docs_2_vec/medical_doc2vec.model"))

    """
    Train word2vec model
    """

    def _word2vec(self, medical_texts):
        # build vocabulary and train model
        model = Word2Vec(
            medical_texts,
            size=150,
            window=5,
            min_count=2,
            workers=10)
        model.train(medical_texts, total_examples=model.corpus_count, epochs=10)
        # save only the word vectors
        model.wv.save(os.path.join(abspath,
                                   "../vectors/words_2_vec/size_150_window_5_mincount_2_german.medical.word2vec.model"))

    def _fastText(self, medical_texts):
        print('fastText')
        model = FastText(sentences=medical_texts, size=150, min_count=2, window=5)
        model.build_vocab(sentences=medical_texts, update=True)
        model.train(sentences=medical_texts, total_examples=len(medical_texts), epochs=7)
        model.wv.save(os.path.join(abspath, "../vectors/fastText/medical.fasttext.model"))


if __name__ == '__main__':
    reader = RawDataReader()
    trainer = TrainerText2Vec()
    docs = reader.get_docs('../german_medical_data_merged/all_merged.gz')
    trainer._doc2vec(docs)
    trainer._fastText(docs)
    trainer._word2vec(docs)
