import sys
import random

import os
from parse import parse
import scipy.sparse as sp
import numpy as np

MOVIES_FILE = 'movies.dat'
RATINGS_FILE = 'ratings.dat'
TRAIN_DIR = 'data'

def filepath(filename):
    return os.path.join(TRAIN_DIR, filename)


with open(filepath(MOVIES_FILE)) as data:
    last_line = data.readlines()[-1].decode()

# num_movies, _ = parse('{}::{}', last_line)
movie_dic = {}

with open(filepath(RATINGS_FILE)) as data:
    for i, line in enumerate(data):
        _, movie, _ = parse('{}::{}::{}', line)
        if movie not in movie_dic:
            movie_dic[movie] = len(movie_dic)
        if i % 10000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

num_movies = len(movie_dic)
print("Number of Movies: " + str(num_movies))


class DataSets:
    def __init__(self, dropout=1):

        self.train, self.test, self.validation = (DataSet(int(num_movies),
                                                          filepath(filename + '.dat'),
                                                          dropout)
                                                  for filename in ('train', 'test', 'validation'))

        with open(filepath(RATINGS_FILE)) as data:
            last_user = None

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

                movies.append(movie_dic[movie])
                ratings.append(float(rating) - 2.5)
                if i % 1000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        print("Loaded data.")
        for dataset in self.test, self.train, self.validation:
            dataset.close_file()



class DataSet:
    def __init__(self, dim, datafile, dropout):
        self.dim = dim
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
        return (sp.csc_matrix((v, (rows, cols)), shape=(batch_size, self.dim)).toarray()
                for v in (values, values_with_dropout))
