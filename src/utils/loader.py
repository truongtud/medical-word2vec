import json
import logging
from os import walk, path


class Loader:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def read_config(self, config_file):
        with open(config_file) as json_config:
            config = json.load(json_config, encoding='utf-8')
        # self.logger.info(config)
        return config

    def list_files(self, directory):
        for (dirpath, dirnames, filenames) in walk(directory):
            return [path.join(dirpath, filename) for filename in filenames]
