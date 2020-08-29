import re
from datetime import datetime


class Utils(object):
    @staticmethod
    def _get_weekday(date_str):
        # print(date_str)
        return datetime.strptime(date_str, '%Y-%m-%d').date().weekday()

    @staticmethod
    def _get_age(age_str):
        expression = '(\d{0,3})(\s?)(year|years)?(\s?)(\d{0,2})(\s?)(mon|mons)?(\s?)(\d{0,3})(\s?)(day|days)?'
        pattern = re.compile(expression)
        matching = pattern.match(age_str)
        return matching.group(1)


if __name__ == '__main__':
    print(Utils._get_weekday('2018-12-14'))
    print(Utils._get_age('55 years 8 mons'))
