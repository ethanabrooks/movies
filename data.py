import argparse
import random
import shutil
import cPickle

import sys

import os
import scipy.sparse as sp
import numpy as np
from progress.bar import IncrementalBar
from parse import parse

DATA_DIR = 'data'
BACKUP_DIR = 'backup'

# names of files where we will pickle and save objects for later use
datasets_file = 'DataObj'
RATINGS = 'ratings.dat'
MOVIE_NAMES = 'movies.dat'
MAX_RATING = 5
MIN_RATING = 0
DIM = 10677
random.seed(7)  # lucky number 7


def progress_bar(message, max):
    return IncrementalBar(message,
                          fill='IncrementalBar',
                          max=max,
                          suffix='%(percent)1.1f%%, ETA: %(eta)ds')


def iterate_if_line1(handle):
    if handle.tell() == 0:
        next(handle)


def num_lines(filepath):
    """ get the number of lines in a file """
    num = 0
    with open(filepath) as fp:
        for line in fp:
            num += 1
    return num


def normalize(rating):
    """ necessary so that the model is not biased toward low ratings """
    return rating - np.mean((MAX_RATING, MIN_RATING))


def unnormalize(rating):
    return rating + np.mean((MAX_RATING, MIN_RATING))


def backup_path(filename):
    """ to be called from a child of the main directory """
    return os.path.join('..', BACKUP_DIR, filename)


def data_path(filename):
    """to be called from the main directory"""
    return os.path.join(DATA_DIR, filename)


def read_cols_vals(line):
    return np.fromstring(line, sep=' ').reshape(2, -1)


class FilePointer:
    def __init__(self, filename, offset):
        self.filename = filename
        self.offset = offset

    def readline(self):
        fp = open(data_path(self.filename), 'r')
        fp.seek(self.offset)
        return fp.readline()


class Data:
    """
    Collector of the train, test, and validation datasets.
    This class creates the other three and contains information common to all.
    """

    def __init__(self, corrupt=1,
                 debug=False,
                 reload=False,
                 ratings=RATINGS,
                 entity_names=MOVIE_NAMES):

        """
        Check if this has already been done. If so, load attributes from file.
        If not, go through the main ratings file, reformat the data, and split
        into train, test, and validation sets.
        """
        # create required dirs if they do not exist
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.isdir(BACKUP_DIR):
            os.mkdir(BACKUP_DIR)
        os.chdir(DATA_DIR)  # this will make other operations easier
        datasets = ['train', 'test', 'validation']

        # these files are the prerequisites for not reprocessing data
        files_that_must_exist = [filename + '.dat' for filename in datasets] + \
                                [datasets_file]

        def data_already_processed():
            return all(os.path.isfile(filepath) and  # it exists
                       os.stat(filepath).st_size != 0  # it isn't empty
                       for filepath in files_that_must_exist)

        # if files are missing, try retrieving from backup
        # if not data_already_processed():
        #     for filename in os.listdir(backup_path('')):
        #         shutil.copyfile(backup_path(filename), filename)

        # check again
        if data_already_processed() and not reload:
            # load self from file
            with open(datasets_file, 'rb') as fp:
                self.__dict__.update(cPickle.load(fp))

        else:  # if data has not already been loaded
            # double check that we want to continue (this wipes existing data)
            # response = raw_input('Are you sure you want to process data? ')
            # while True:
            #     if response in 'Yes yes':
            #         break
            #     elif response in 'No no':
            #         print('Ok. Goodbye.')
            #         exit(0)
            #     else:
            #         response = raw_input('Please enter [y|n]. ')

            if debug:
                ratings = 'debug.dat'

            # create the three datasets
            self.datasets = []
            for name in datasets:
                dataset = DataSet(name + '.dat', corrupt)
                self.__dict__[name] = dataset
                self.datasets.append(dataset)

            # id_to_column assigns each movie to a column
            # such that all columns are contiguous
            self.id_to_column = {}
            self.user_dic = {}

            # convert data into a more usable form
            # and split into train, validation, and test
            bar = progress_bar('Loading ratings data', num_lines(ratings))
            with open(ratings) as data:
                while True:
                    parsed = self.parse_data(data, bar)
                    if parsed is None:
                        break

                    user, movies, ratings = parsed

                    for i, movie in enumerate(movies):
                        # this is how id_to_column ensures contiguous columns
                        if movie not in self.id_to_column:
                            self.id_to_column[movie] = len(self.id_to_column)
                        movies[i] = self.id_to_column[movie]

                    self.write_instance(user, movies, ratings)

                bar.finish()

            print("Loaded data.")

            for dataset in self.datasets:
                dataset.file_handle.close()

            # the dimension of each instance
            self.dim = len(self.id_to_column)
            for dataset in self.datasets:
                dataset.set_dim(self.dim)

            with open(entity_names) as datafile:
                self.name_to_column, self.column_to_name = self.populate_dicts(datafile)  # save self to file
            with open(datasets_file, 'w') as fp:
                cPickle.dump(self.__dict__, fp, 2)

            # backup
            for filename in files_that_must_exist:
                shutil.copyfile(filename, backup_path(filename))

        os.chdir('..')  # return to main dir

    def populate_dicts(self, handle):
        name_to_column = {}
        column_to_name = {}
        for line in handle:
            id, name, _ = parse('{:d}::{} ({}', line)
            if id in self.id_to_column:
                movies = self.id_to_column[id]
                name_to_column[name] = movies
                column_to_name[movies] = name
        return name_to_column, column_to_name

    def parse_data(self, data, bar):
        last_user = None
        movies, ratings = ([] for _ in range(2))
        for i, line in enumerate(data):
            user, movie, rating, _ = parse('{:d}::{:d}::{:g}:{}', line)
            if user != last_user:  # if we're on to a new user
                if last_user is not None:
                    return last_user, movies, ratings

                # clean slate for next user
                movies, ratings = ([] for _ in range(2))
                last_user = user

            movies.append(movie)
            ratings.append(normalize(rating))

            # progress bar
            bar.next()

    def write_instance(self, user, movies, ratings):
        random_num = random.random()
        if random_num < .7:  # 70% of the rime
            dataset = self.train
        elif random_num < .9:  # 20% of the time
            dataset = self.test
        else:  # 10% of the time
            dataset = self.validation

        # write instance to the file associated with the dataset
        pos = dataset.new_instance(movies, ratings)

        # save a "pointer" to the users position in the file
        self.user_dic[user] = FilePointer(dataset.datafile, pos)

    def get_col(self, movie):
        """
        :param movie_id from preprocessed data
        :returns column number in instance array
        """
        return self.name_to_column[movie]

    def get_ratings(self, user_id):
        """
        :returns the ratings instance corresponding to the user_id
        (for feeding into the model)
        """
        if user_id not in self.user_dic:
            print('Sorry, no user with that id.')
            exit(0)
        line = self.user_dic[user_id].readline()
        cols, values = read_cols_vals(line)
        rows = np.zeros_like(cols)
        return sp.csc_matrix((values, (rows, cols)),
                             dtype='float32',
                             shape=[1, self.dim]).toarray()


class DataSet:
    def __init__(self, datafile, corrupt):
        self.datafile = datafile

        # we leave the file_handle open for speed
        self.file_handle = open(self.datafile, 'w')
        self.corrupt = corrupt
        self.num_examples = 0

    def new_instance(self, movies, ratings):
        """
        Write movies and ratings to the file associated with this dataset.
        This data will later be read from the file during training.
        :returns the position where the data was saved
        """
        self.num_examples += 1
        data = np.r_[movies, ratings].reshape(1, -1)
        pos = self.file_handle.tell()
        np.savetxt(self.file_handle, data, fmt='%1.1f')
        return pos

    def set_dim(self, dim):
        self.dim = dim

    def next_batch(self, batch_size):
        """
        This method is called repeatedly during training to retrieve
        the next batch of training data
        """
        # These values will later be used to construct a sparse matrix
        values, rows, cols = ([] for _ in range(3))

        if self.file_handle.closed:
            self.file_handle = open(os.path.join(DATA_DIR, self.datafile), 'r')
            self.file_handle.seek(0)

        for i, line in enumerate(self.file_handle):
            try:  # in case we hit a blip, we don't want to terminate training
                movies, ratings = read_cols_vals(line)
                values.append(ratings)
                cols.append(movies)
                rows.append(np.repeat(i, movies.size))
            except ValueError:  # just skip the data point
                pass

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
        corrupted_values = values.copy()
        idxs_to_corrupt = np.random.choice(values.size, self.corrupt, replace=False)
        corrupted_values[idxs_to_corrupt] = 0

        # is_data_mask will later be used to mask missing values so that our accuracy
        # scores are based only on the ratings that were actually present in the dataset
        # (otherwise all the unrated movies would give our model an unfairly good score)
        inputs, targets, is_data_mask = (
            sp.csc_matrix((vals, (rows, cols)), shape=(batch_size, self.dim)).toarray()
            for vals in (values, corrupted_values, np.ones_like(values)))
        return inputs, targets, is_data_mask


if __name__ == '__main__':
    import data

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    os.chdir('EasyMovies')

    data.Data(debug=args.debug, reload=args.reload)
