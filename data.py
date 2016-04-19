import argparse
import random
import shutil
import cPickle

import sys

import abc
import os
import scipy.sparse as sp
import numpy as np
from progress.bar import IncrementalBar
from parse import parse

DATA_DIR = 'data'
BACKUP_DIR = 'backup'
DEBUG_DIR = 'debug'

# names of files where we will pickle and save objects for later use
DATA_OBJ = 'DataObj'
RATINGS = 'ratings.dat'
MOVIE_NAMES = 'movies.dat'
DEBUG_FILE = 'debug.dat'
MAX_RATING = 5
MIN_RATING = 0
DEBUG_STOP = 3000
DIM = 10677
DATASET_NAMES = 'train test validation'.split()
FILES_THAT_MUST_EXIST = [name + '.dat' for name in DATASET_NAMES] + [DATA_OBJ]
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


def read_cols_vals(line):
    return np.fromstring(line, sep=' ').reshape(2, -1)


def empty(filepath):
    return os.stat(filepath).st_size == 0


class FilePointer:
    def __init__(self, root, filename, offset):
        self.root = root
        self.filename = filename
        self.offset = offset

    def readline(self):
        with open(os.path.join(self.root, self.filename), 'r') as fp:
            fp.seek(self.offset)
            return fp.readline()


class Data:
    """
    Collector of the train, test, and validation datasets.
    This class creates the other three and contains information common to all.
    """
    one_and_only = None
    __metaclass__ = abc.ABCMeta

    def __init__(self, corrupt=1,
                 debug=False,
                 ratings=RATINGS,
                 entity_names=MOVIE_NAMES,
                 load_previous=False):

        """
        Check if this has already been done. If so, load attributes from file.
        If not, go through the main ratings file, reformat the data, and split
        into train, test, and validation sets.
        """
        self.debug = debug

        # There. Singleton.
        if Data.one_and_only is None:
            Data.one_and_only = self
        else:
            self.__dict__.update(Data.one_and_only)
            return

        if load_previous:
            self.load_previous()
            return

        # create the three datasets
        self.datasets = []
        for name in DATASET_NAMES:
            dataset = DataSet(name + '.dat', corrupt, debug)
            self.__dict__[name] = dataset
            self.datasets.append(dataset)

        # id_to_column assigns each movie to a column
        # such that all columns are contiguous
        self.id_to_emb_idx = {}
        self.user_dic = {}

        # convert data into a more usable form
        # and split into train, validation, and test
        ratings_file = os.path.join(DATA_DIR, ratings)
        nlines = DEBUG_STOP if debug else num_lines(ratings_file)
        bar = progress_bar('Loading ratings data', nlines)
        with open(ratings_file) as data:
            while True:
                parsed = self.parse_data(data, bar)
                if parsed is None:
                    break

                user, entities, ratings = parsed
                for i, entity in enumerate(entities):

                    # this is how id_to_emb_idx ensures contiguous columns
                    if entity not in self.id_to_emb_idx:
                        self.id_to_emb_idx[entity] = len(self.id_to_emb_idx)
                    entities[i] = self.id_to_emb_idx[entity]
                self.write_instance(user, entities, ratings)
                if debug and len(self.id_to_emb_idx) > DEBUG_STOP:
                    print('\nStop early for debug.')
                    break

        bar.finish()
        print('Close file handles')
        self.close_file_handles()

        print("Loaded data.")

        # the number of different movies/books/entities
        self.emb_size = len(self.id_to_emb_idx) + 1
        for dataset in self.datasets:
            dataset.set_emb_size(self.emb_size)

        # get dicts that translate between string names and columns in the large
        # sparse data vector (one element per entity)
        with open(os.path.join(DATA_DIR, entity_names)) as datafile:
            self.name_to_column, self.column_to_name = self.populate_dicts(datafile)  # save self to file

        # save self to file
        root = DEBUG_DIR if debug else DATA_DIR
        with open(os.path.join(root, DATA_OBJ), 'w') as fp:
            cPickle.dump(self.__dict__, fp, 2)

        # check that essential files didn't somehow get deleted
        paths = [os.path.join(root, filename) for filename in
                 [name + '.dat' for name in DATASET_NAMES] + [DATA_OBJ]]
        for filepath in paths:
            assert not empty(filepath)
        print('Data made it safely to file :)')

    def close_file_handles(self):
        for dataset in self.datasets:
            dataset.file_handle.close()

    def backup(self, files_that_must_exist):
        for filename in files_that_must_exist:
            shutil.copyfile(filename, os.path.join(BACKUP_DIR, filename))

    @abc.abstractmethod
    def populate_dicts(self, handle):
        """
        :param handle a data handle in some file that correlates entity names
        with numerical entity ids
        :returns two dictionaries (name_to_column, column_to_name), where name
        is the string name of the entity and 'column' is the location where that
        entity is going to end up on the big sparse ratings vector
        """
        # name_to_column = {}
        # column_to_name = {}
        # for line in handle:
        #     id, name, _ = parse('{:d}::{} ({}', line)
        #     if id in self.id_to_emb_idx:
        #         movies = self.id_to_emb_idx[id]
        #         name_to_column[name] = movies
        #         column_to_name[movies] = name
        # return name_to_column, column_to_name

    @abc.abstractmethod
    def parse_data(self, data, bar):
        """
        :param data: a handle in a data file
        :param bar: a purty loading bar
        parse_data is responsible for iterating both of these
        :returns a (entities, ratings) tuple (not-interleaved)
        """
        # last_user = None
        # movies, ratings = ([] for _ in range(2))
        # for i, line in enumerate(data):
        #     user, movie, rating, _ = parse('{:d}::{:d}::{:g}:{}', line)
        #     if user != last_user:  # if we're on to a new user
        #         if last_user is not None:
        #             return last_user, movies, ratings
        #
        #         # clean slate for next user
        #         movies, ratings = ([] for _ in range(2))
        #         last_user = user
        #
        #     movies.append(movie)
        #     ratings.append(normalize(rating))
        #
        #     # progress bar
        #     bar.next()

    def load_previous(self):
        """
        :param files_that_must_exist these files are the
        prerequisites for not reprocessing data
        """
        paths = (os.path.join(DATA_DIR, name)
                 for name in FILES_THAT_MUST_EXIST)
        for filepath in paths:
            if not os.path.isfile(filepath):
                print(filepath + ' does not exist.')
                exit(0)
            if empty(filepath):
                print(filepath + ' is empty.')
                exit(0)

        # load hibernating clone from file
        with open(os.path.join(DATA_DIR, DATA_OBJ), 'rb') as fp:
            self.__dict__.update(cPickle.load(fp))

    def write_instance(self, user, entities, ratings):
        random_num = random.random()
        if random_num < .7:  # 70% of the rime
            dataset = self.train
        elif random_num < .9:  # 20% of the time
            dataset = self.test
        else:  # 10% of the time
            dataset = self.validation

        # write instance to the file associated with the dataset
        pos = dataset.new_instance(entities, ratings)

        # save a "pointer" to the users position in the file
        path = DEBUG_DIR if self.debug else DATA_DIR
        self.user_dic[user] = FilePointer(path, dataset.datafile, pos)

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
                             shape=[1, self.emb_size]).toarray()


class DataSet:
    def __init__(self, datafile, corrupt, debug):
        self.datafile = datafile

        # we leave the file_handle open for speed
        data_dir = DEBUG_DIR if debug else DATA_DIR
        self.file_handle = open(os.path.join(data_dir, datafile), 'w')
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

    def set_emb_size(self, emb_size):
        self.emb_size = emb_size

    def next_batch(self, batch_size):
        """
        This method is called repeatedly during training to retrieve
        the next batch of training data
        """
        # These values will later be used to construct a sparse matrix
        values, rows, cols = ([] for _ in range(3))

        filepath = os.path.join(DATA_DIR, self.datafile)
        if self.file_handle.closed:
            self.file_handle = open(filepath, 'r')
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
            if empty(filepath):
                print('The data file is empty')
                exit(0)
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
            sp.csc_matrix((vals, (rows, cols)), shape=(batch_size, self.emb_size)).toarray()
            for vals in (values, corrupted_values, np.ones_like(values)))
        return inputs, targets, is_data_mask


def assert_exists(filepath):
    assert os.path.isfile(filepath), '{0}/{1} not found'.format(os.getcwd(), filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    RATINGS = 'ratings.dat'
    MOVIE_NAMES = 'movies.dat'
    DIR = 'debug' if args.debug else 'data'
    os.chdir('EasyMovies')
    files_that_must_exist = (os.path.join(DATA_DIR, name)
                             for name in (RATINGS, MOVIE_NAMES))
    for filepath in files_that_must_exist:
        assert_exists(filepath)

    try:
        Data(debug=args.debug)
    except IOError as error:
        print('cwd: ' + os.getcwd())
        print(error)
