import random
import shutil
import cPickle

import os
from parse import parse
import scipy.sparse as sp
import numpy as np
from progress.bar import IncrementalBar

TRAIN_DIR = 'data'
# names of files where we will pickle and save objects for later use
datasets_file = 'DataSets'
movie_dic_file = 'movie_dic'


def progress_bar(message, max):
    return IncrementalBar(message,
                          fill='IncrementalBar',
                          max=max,
                          suffix='%(percent)1.1f%%, ETA: %(eta)ds')


def num_lines(filepath):
    """ get the number of lines in a file """
    num = 0
    with open(filepath) as fp:
        for line in fp:
            num += 1
    return num



def backup_path(filename):
    return os.path.join(os.path.join('..', 'backup'), filename)


class DataSets:
    """
    Collector of the train, test, and validation datasets.
    This class creates the other three and contains information common to all.
    """

    def __init__(self, corrupt=1, datafile='ratings.dat'):
        """
        Check if this has already been done. If so, load attributes from file.
        If not, go through the main ratings file, reformat the data, and split
        into train, test, and validation sets.
        """
        if not os.path.isdir(TRAIN_DIR):
            os.mkdir(TRAIN_DIR)  # create the train dir if it does not exist
        os.chdir(TRAIN_DIR)  # this will make other operations easier

        datasets = ['train', 'test', 'validation']

        # these files are the prerequisites for not reprocessing data
        files_that_must_exist = [filename + '.dat' for filename in datasets] + \
                                [datasets_file, movie_dic_file]

        def data_already_processed():
            return all(os.path.isfile(filepath) and  # it exists
                       os.stat(filepath).st_size != 0  # it isn't empty
                       for filepath in files_that_must_exist)

        # if files are missing, try retrieving from backup
        if not data_already_processed():
            backup = os.path.join('..', 'backup')
            for filename in os.listdir(backup):
                shutil.copyfile(backup_path(filename), filename)

        # check again
        if data_already_processed():
            # load self from file
            with open(datasets_file, 'rb') as fp:
                self.__dict__.update(cPickle.load(fp))

        else:  # if data has not already been loaded

            # create datasets attribute
            self.datasets = []
            for name in datasets:
                dataset = DataSet(name + '.dat', corrupt)
                self.__dict__[name] = dataset
                self.datasets.append(dataset)

            # movie_dic assigns a unique id to each movie such that all ids are contiguous
            movie_dic = {}
            with open(datafile) as data:
                last_user = None
                bar = progress_bar('Loading ratings data', num_lines(datafile))
                for i, line in enumerate(data):
                    user, movie, rating, _ = parse('{}::{}::{}::{}', line)
                    if user != last_user:  # if not first line of file
                        if last_user is not None:
                            random_num = random.random()
                            if random_num < .7:  # 70% of the rime
                                name = self.train
                            elif random_num < .9:  # 20% of the time
                                name = self.test
                            else:  # 10% of the time
                                name = self.validation
                            name.new_instance(movies, ratings)
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

            # save self to file
            with open(datasets_file, 'w') as fp:
                cPickle.dump(self.__dict__, fp, 2)

            # save movie dictionary to file
            with open(movie_dic_file, 'w') as fp:
                cPickle.dump(movie_dic, fp, 2)

            # backup
            for filename in files_that_must_exist:
                shutil.copyfile(filename, backup_path(filename))

        os.chdir('..')  # return to main dir

    def set_data_dim(self, dim):
        for dataset in self.datasets:
            self.dim = dataset.dim = dim
            dataset.file_handle.close()

    def get_movie(self, movie_id):
        movie_dic = cPickle.load(movie_dic_file)
        return movie_dic[movie_id]


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
        """
        self.num_examples += 1
        data = np.r_[movies, ratings].reshape(1, -1)
        np.savetxt(self.file_handle, data, fmt='%1.1f')

    def next_batch(self, batch_size):
        """
        This method is called repeatedly during training to retrieve
        the next batch of training data
        """
        # These values will later be used to construct a sparse matrix
        values, rows, cols = ([] for _ in range(3))

        if self.file_handle.closed:
            self.file_handle = open(os.path.join(TRAIN_DIR, self.datafile), 'r')
            self.file_handle.seek(0)

        for i, line in enumerate(self.file_handle):
            fromstring = np.fromstring(line, sep=' ')
            try:  # in case we hit a blip, we don't want to terminate training
                movies, ratings = fromstring.reshape(2, -1)
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


def get_ratings(userid):
    return None
