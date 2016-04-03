import argparse
import csv
import re
import sys

import os
import data
from data import Data
from parse import parse

RATINGS = 'BX-Book-Ratings.csv'
MOVIE_NAMES = 'BX-Books.csv'


class Books(Data):
    def __init__(self, ratings=RATINGS, entity_names=MOVIE_NAMES, debug=False, reload=True):
        Data.__init__(self, ratings=ratings, entity_names=entity_names, debug=debug, reload=reload)

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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    os.chdir('Books')
    Books(debug=args.debug, reload=args.reload)
