import argparse
import csv
import re
import sys

import os
import data
from data import Data
from parse import parse

RATINGS = 'ratings.csv'
MOVIE_NAMES = 'movies.csv'
DIM = 10677


class Movies(Data):
    def __init__(self, ratings=RATINGS, entity_names=MOVIE_NAMES, debug=False, load_previous=False):
        Data.__init__(self, ratings=ratings, entity_names=entity_names, debug=debug, load_previous=load_previous)

    def parse_data(self, handle, bar):
        data.iterate_if_line1(handle)
        reader = csv.reader(handle)
        last_user = None
        for line in reader:
            #progress bar
            bar.next()
            user, movie, rating, _ = line
            user, movie = map(int, (user, movie))
            rating = float(rating)
            if user != last_user:  # if we're on to a new user
                if last_user is not None:
                    return last_user, movies, values

                # clean slate for next user
                movies, values = ([] for _ in range(2))
                last_user = user

            movies.append(movie)
            values.append(rating)
        bar.next()

    def populate_dicts(self, handle):
        data.iterate_if_line1(handle)
        name_to_id, id_to_name = {}, {}
        reader = csv.reader(handle)
        for line in reader:
            id, name, _ = line
            id = int(id)
            name = re.match(r'(.*?)(\s\(|$)', name).group(1)
            id_to_name[id] = name
            name_to_id[name] = id
        return name_to_id, id_to_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    os.chdir('Movies')
    files_that_must_exist = (os.path.join(data.DATA_DIR, name)
                             for name in (RATINGS, MOVIE_NAMES))
    for filepath in files_that_must_exist:
        data.assert_exists(filepath)

    try:
        Movies(debug=args.debug)
    except OSError as error:
        print('cwd: ' + os.getcwd())
        print(error)
