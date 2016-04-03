import argparse
import csv
import re
import sys

import os
import data
from data import Data
from parse import parse

RATINGS = 'BX-Book-Ratings.csv'
BOOK_NAMES = 'BX-Books.csv'


class Books(Data):
    def __init__(self, ratings=RATINGS, entity_names=BOOK_NAMES, debug=False, load_previous=False):
        Data.__init__(self, ratings=ratings, entity_names=entity_names, debug=debug, load_previous=load_previous)

    def parse_data(self, handle, bar):
        data.iterate_if_line1(handle)
        reader = csv.reader(handle, delimiter=';')
        last_user = None
        for line in reader:
            user, book, rating = line
            user = int(user)
            rating = int(rating)
            if user != last_user:  # if we're on to a new user
                if last_user is not None:
                    return last_user, books, ratings

                # clean slate for next user
                books, ratings = ([] for _ in range(2))
                last_user = user

            books.append(book)
            ratings.append(rating)
            bar.next()

    def populate_dicts(self, handle):
        data.iterate_if_line1(handle)
        name_to_id, id_to_name = {}, {}
        reader = csv.reader(handle, delimiter=';')
        for line in reader:
            id, name = line[:2]
            name = re.match(r'(.*?)(\s\(|$)', name).group(1)
            id_to_name[id] = name
            name_to_id[name] = id
        return name_to_id, id_to_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    os.chdir('Books')
    files_that_must_exist = (os.path.join(data.DATA_DIR, name)
                             for name in (RATINGS, BOOK_NAMES))
    for filepath in files_that_must_exist:
        data.assert_exists(filepath)

    try:
        Books(debug=args.debug)
    except IOError as error:
        print('cwd: ' + os.getcwd())
        print(error)
