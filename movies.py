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
    def __init__(self, data_file=RATINGS, vocab=MOVIE_NAMES):
        Data.__init__(self, data_file, vocab)

    def parse_data(self, handle):
        data.iterate_if_line1(handle)
        reader = csv.reader(handle)
        last_user = None
        for line in reader:
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

    os.chdir('Movies')
    if len(sys.argv) > 1:
        if sys.argv[1] == 'debug':
            Movies(debug=True)
        elif sys.argv[1] == 'reload':
            Movies(reload=True)
    else:
        Movies()
