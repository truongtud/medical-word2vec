import logging

from stop_words import safe_get_stop_words


class Cleaner:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.stop_words = safe_get_stop_words('german')
        print(self.stop_words)

    def clean(self, text):
        return self.replace_special_characters(self.replace_umlauts(text))

    def is_stop_word(self, word):
        return word in self.stop_words

    def replace_umlauts(self, text):
        """
        Replaces german umlauts and sharp s in given text.
        :param text: text as str
        :return: manipulated text as str
        """
        res = text
        res = res.replace('ä', 'ae')
        res = res.replace('ö', 'oe')
        res = res.replace('ü', 'ue')
        res = res.replace('Ä', 'Ae')
        res = res.replace('Ö', 'Oe')
        res = res.replace('Ü', 'Ue')
        res = res.replace('ß', 'ss')
        return res

    def replace_special_characters(self, text):
        # print(text)
        import re
        rx = re.compile('([{}()])')
        text = text.lower()
        # text = text.replace('_', ' ')
        # text=text.replace('-', ' ')
        text = rx.sub(' ', text)
        rx = re.compile('([";,:-@#$~&])')
        text = rx.sub(' ', text)
        return text
