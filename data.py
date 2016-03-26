import sys
import random

import os
from parse import parse
import scipy.sparse as sp
import numpy as np
from progress.bar import Bar, IncrementalBar

MOVIES_FILE = 'movies.dat'
LEN_MOVIES_FILE = 10681
RATINGS_FILE = 'ratings.dat'
LEN_RATINGS_FILE = 10000054
TRAIN_DIR = 'data'


def progress_bar():
    return IncrementalBar('Loading ratings data',
                          fill='IncrementalBar',
                          max=LEN_RATINGS_FILE,
                          suffix='%(percent)1.1f%%, ETA: %(eta)ds')


def filepath(filename):
    return os.path.join(TRAIN_DIR, filename)


class DataSets:
    def __init__(self, dropout=1):

        self.train, self.test, self.validation = (DataSet(filepath(filename + '.dat'),
                                                          dropout)
                                                  for filename in ('train', 'test', 'validation'))

        movie_dic = {}
        with open(filepath(RATINGS_FILE)) as data:
            last_user = None

            bar = progress_bar()
            for i, line in enumerate(data):
                user, movie, rating, _ = parse('{}::{}::{}::{}', line)

                if user != last_user:  # if not first line
                    if last_user is not None:
                        random_num = random.random()
                        if random_num < .7:
                            dataset = self.train
                        elif random_num < .9:  # 20% of the time
                            dataset = self.test
                        else:  # 10% of the time
                            dataset = self.validation
                        dataset.new_instance(movies, ratings)

                    movies, ratings = ([] for _ in range(2))
                    last_user = user

                if movie not in movie_dic:
                    movie_dic[movie] = len(movie_dic)
                movies.append(movie_dic[movie])
                ratings.append(float(rating) - 2.5)
                bar.next()
            bar.finish()

        print("Loaded data.")
        for dataset in self.test, self.train, self.validation:
            self.dim = dataset.dim = len(movie_dic)
            dataset.close_file()


class DataSet:
    def __init__(self, datafile, dropout):
        self.datafile = datafile
        self.file_handle = open(datafile, 'w')
        self.dropout = dropout
        self.num_examples = 0

    def new_instance(self, movies, ratings):
        self.num_examples += 1
        data = np.r_[movies, ratings].reshape(1, -1)
        np.savetxt(self.file_handle, data, fmt='%1.1f')

    def close_file(self):
        self.file_handle.close()

    def next_batch(self, batch_size):
        values, rows, cols = ([] for _ in range(3))

        if self.file_handle.closed:
            self.file_handle = open(self.datafile, 'r')

        for i, line in enumerate(self.file_handle):
            movies, ratings = np.fromstring(line, sep=' ').reshape(2, -1)
            values.append(ratings)
            cols.append(movies)
            rows.append(np.repeat(i, movies.size))

            if i == batch_size - 1:
                break

        if not values:  # if handle was at the end of the file
            self.file_handle.close()
            return self.next_batch(batch_size)

        values, cols, rows, = (np.hstack(l) for l in (values, cols, rows))
        values_with_dropout = values.copy()
        idxs_to_dropout = np.random.choice(values.size, self.dropout, replace=False)
        values_with_dropout[idxs_to_dropout] = 0
        # inputs, targets, is_data_mask = (
        #     sp.csc_matrix((vals, (rows, cols)), shape=(batch_size, self.dim))
        #     for vals in (values, values_with_dropout, np.ones_like(values)))
        return values, values_with_dropout, cols, rows
