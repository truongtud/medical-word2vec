import gzip
import logging
import os

import gensim

from utils.cleaner import Cleaner
from utils.loader import Loader

abspath = os.path.dirname(os.path.abspath(__file__))


class RawDataReader:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.cleaner = Cleaner()
        self.loader = Loader()

    def read_from_gz(self, input_file):
        logging.info("reading file {0}...this may take a while".format(input_file))
        with gzip.open(input_file, 'rt', errors='ignore', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = self.cleaner.clean(line)
                if (i % 20000 == 0):
                    logging.info("read {0} medical text".format(i))
                    print(line)
                # setting max_len is important to avoid missing words
                yield gensim.utils.simple_preprocess(line, max_len=5000)

    def get_docs_from_gz(self, gz_file):
        # data_file = os.path.join(abspath, gzip)
        data_file = gz_file
        documents = list(self.read_from_gz(data_file))
        logging.info("Done reading data file")
        return documents

    def get_docs_from_file(self, text_file):
        with open(text_file, 'r', errors='ignore') as file:
            yield file.data()

    def get_docs_from_directory(self, directory):
        files = self.loader.list_files(directory)
        documents = list(self.get_docs_from_file(file) for file in files)
        return documents

    def get_docs(self, path):
        import pathlib
        print(pathlib.Path(path).suffixes)
        if '.gz' in pathlib.Path(path).suffixes:
            return self.get_docs_from_gz(path)
        elif os.path.isfile(path):
            return self.get_docs_from_file(path)
        else:
            """ Read files from directory"""
            return self.get_docs_from_directory(path)
