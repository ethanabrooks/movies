import argparse
import csv
import re
import sys

import os
import data
from data import normalize
from data import Data
from parse import parse

RATINGS = 'ratings.dat'
MOVIE_NAMES = 'movies.dat'
DIM = 10677


class EasyMovies(Data):
    def __init__(self, ratings=RATINGS, entity_names=MOVIE_NAMES, debug=False, load_previous=False):
        Data.__init__(self, ratings=ratings, entity_names=entity_names, debug=debug, load_previous=load_previous)

    def parse_data(self, data, bar):
        last_user = None
        movies, ratings = ([] for _ in range(2))
        for i, line in enumerate(data):
            # progress bar
            bar.next()
            user, movie, rating, _ = parse('{:d}::{:d}::{:g}:{}', line)
            if user != last_user:  # if we're on to a new user
                if last_user is not None:
                    return last_user, movies, ratings

                # clean slate for next user
                movies, ratings = ([] for _ in range(2))
                last_user = user

            movies.append(movie)
            ratings.append(normalize(rating))
        bar.next()


    def populate_dicts(self, handle):
        name_to_column = {}
        column_to_name = {}
        for line in handle:
            id, name, _ = parse('{:d}::{} ({}', line)
            if id in self.id_to_emb_idx:
                movies = self.id_to_emb_idx[id]
                name_to_column[name] = movies
                column_to_name[movies] = name
        return name_to_column, column_to_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    os.chdir('EasyMovies')
    files_that_must_exist = (os.path.join(data.DATA_DIR, name)
                             for name in (RATINGS, MOVIE_NAMES))
    for filepath in files_that_must_exist:
        data.assert_exists(filepath)

    try:
        EasyMovies(debug=args.debug)
    except OSError as error:
        print('cwd: ' + os.getcwd())
        print(error)
