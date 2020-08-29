import logging
import os

import nltk
from gensim.models import KeyedVectors as kv
from gensim.models.doc2vec import Doc2Vec

nltk.download('punkt')

abspath = os.path.dirname(os.path.abspath(__file__))


class ModelReader():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_word2vec_model(self, model_path):
        model_file = os.path.join(abspath, model_path)
        model = kv.load(model_file)
        return model

    def get_doc2vec_model(self, model_path):
        p_model = os.path.join(abspath, model_path)
        model = Doc2Vec.load(p_model)
        return model

    def get_fasttext_model(self, model_path):
        model_file = os.path.join(abspath, model_path)
        model = kv.load(model_file)
        return model

    def get_model(self, model_path, model_type):
        if model_type == 'doc2vec':
            return self.get_doc2vec_model(model_path)
        elif model_type == 'word2vec':
            return self.get_word2vec_model(model_path)
        elif model_type == 'fasttext':
            return self.get_fasttext_model(model_path)
