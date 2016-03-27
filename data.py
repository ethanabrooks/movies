import sys
import random

import os
from parse import parse
import scipy.sparse as sp
import numpy as np
from progress.bar import Bar, IncrementalBar

MOVIES_FILE = 'movies.dat'
LEN_MOVIES_FILE = 10681
LEN_RATINGS_FILE = 10000054
TRAIN_DIR = 'data'


def progress_bar(message):
    return IncrementalBar(message,
                          fill='IncrementalBar',
                          max=LEN_RATINGS_FILE,
                          suffix='%(percent)1.1f%%, ETA: %(eta)ds')


def filepath(filename):
    return os.path.join(TRAIN_DIR, filename)


class DataSets:
    def __init__(self, dropout=1, datafile='ratings.dat'):
        self.train, self.test, self.validation = (
            DataSet(filepath(filename + '.dat'), dropout)
            for filename in ('train', 'test', 'validation'))

        # this empty file will indicate that data has already been loaded
        data_loaded = filepath('.data_loaded')

        os.remove(data_loaded)  #TODO: for debugging only!!!

        if os.path.isfile(data_loaded):
            with open(data_loaded) as fp:
                self.set_data_dim(int(fp.readline(1)))

        else:  # if data has not already been loaded
            # movie_dic assigns a unique id to each movie such that all ids are contiguous
            movie_dic = {}
            with open(filepath(datafile)) as data:
                last_user = None
                bar = progress_bar('Loading ratings data')
                for i, line in enumerate(data):
                    user, movie, rating, _ = parse('{}::{}::{}::{}', line)
                    if user != last_user:  # if not first line of file
                        if last_user is not None:
                            random_num = random.random()
                            if random_num < .7:  # 70% of the rime
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

            # set dim attribute for all datasets
            self.set_data_dim(len(movie_dic))

            # create file to indicate that data has been loaded
            with open(data_loaded, 'w') as fp:
                fp.write(str(self.dim))

    def set_data_dim(self, dim):
        for dataset in self.test, self.train, self.validation:
            self.dim = dataset.dim = dim
            dataset.close_file()


class DataSet:
    def __init__(self, datafile, dropout):
        self.datafile = datafile
        self.file_handle = open(datafile, 'w')  # we leave the file_handle open for speed
        self.dropout = dropout
        self.num_examples = 0

    def new_instance(self, movies, ratings):
        """
        Write movies and ratings to the file associated with this dataset.
        This data will later be read from the file during training.
        """
        self.num_examples += 1
        data = np.r_[movies, ratings].reshape(1, -1)
        np.savetxt(self.file_handle, data, fmt='%1.1f')

    def close_file(self):
        """
        DataSets closes the file handles of each of the three datasets
        after the data has been loaded
        """
        self.file_handle.close()

    def next_batch(self, batch_size):
        """
        This method is called repeatedly during training to retrieve
        the next batch of training data
        """
        # These values will later be used to construct a sparse matrix
        values, rows, cols = ([] for _ in range(3))

        if self.file_handle.closed:
            self.file_handle = open(self.datafile, 'r')
            self.file_handle.seek(0)

        for i, line in enumerate(self.file_handle):
            movies, ratings = np.fromstring(line, sep=' ').reshape(2, -1)
            values.append(ratings)
            cols.append(movies)
            rows.append(np.repeat(i, movies.size))

            if i == batch_size - 1:
                break

        if not values:  # if handle was at the end of the file
            self.file_handle.close()
            return self.next_batch(batch_size)  # restart at the beginning of the file

        # At this point values, cols, and rows are lists of (n,) shape arrays
        # We now concatenate them in preparation for creating a sparse matrix
        values, cols, rows, = (np.hstack(l) for l in (values, cols, rows))

        # In our target data, we selectively zero out certain data so the model
        # has to learn to reconstruct the missing values based on those that remain
        values_with_dropout = values.copy()
        idxs_to_dropout = np.random.choice(values.size, self.dropout, replace=False)
        values_with_dropout[idxs_to_dropout] = 0

        # is_data_mask will later be used to mask missing values so that our accuracy
        # scores are based only on the ratings that were actually present in the dataset
        # (otherwise all the unrated movies would give our model an unfairly good score)
        inputs, targets, is_data_mask = (
            sp.csc_matrix((vals, (rows, cols)), shape=(batch_size, self.dim)).toarray()
            for vals in (values, values_with_dropout, np.ones_like(values)))
        return inputs, targets, is_data_mask
