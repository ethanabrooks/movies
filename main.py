import argparse

import numpy as np
import os
import data

def check_if_ok_to_continue(prompt):
    response = raw_input(prompt)
    while True:
        if response in 'Yes yes':
            return
        elif response in 'No no':
            print('Ok. Goodbye.')
            exit(0)
        else:
            response = raw_input('Please enter [y|n]. ')


def prompt_for_int(prompt):
    response = raw_input(prompt)
    while True:
        try:
            return int(response)
        except ValueError:
            response = raw_input('Please enter an integer.')


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='train model from scratch')
parser.add_argument('-u', '--user_id', type=int, help='user to predict ratings for')
parser.add_argument('-m', '--movie', type=str, help='movie to predict ratings for')
parser.add_argument('-t', '--top', type=int, help='the top [n] highest rated movies'
                                                  'predicted for user')
args = parser.parse_args()

# import statement down here so that command line args aren't intercepted by tf.flags
import model

# check if user has entered both a username and a movie title
if not args.movie and not args.top:
    args.movie = raw_input('Please enter the movie you would like '
                           'to predict ratings for: ')
if not args.user_id:
    args.user_id = prompt_for_int('Please enter the user whose rating of %s '
                                  'you would like to predict: ' % args.movie)

# path to saved version of trained model
load_path = os.path.join('checkpoints', 'checkpoint')

# check if a model has been previously trained
already_trained = os.path.exists(load_path)
if not (args.train or already_trained):
#     check_if_ok_to_continue('Model has not been trained. '
#                             'Train it now (this may take several hours)? ')
    args.train = True

if args.train:
    try:
        model.run_training()
    except KeyboardInterrupt:
        print('\nGoodbye')
        exit(0)

# predict a rating for the user
if args.user_id and (args.movie or args.top):
    dat = data.Data()
    instance = dat.get_ratings(args.user_id)
    ratings = data.unnormalize(instance.ravel())
    output = model.predict(instance, dat).ravel()
    if args.movie:
        col = dat.get_col(args.movie)
        rating = output[col]

        # purty stars
        num_stars = int(round(rating * 2))
        stars = ''.join(u'\u2605' for _ in range(num_stars))
        stars += ''.join(u'\u2606' for _ in range(10 - num_stars))

        print("The model predicts that user %s will rate "
              "movie number %s: "
              % (args.user_id, args.movie))
        print('%1.2f / 5' % rating)
        print(stars)
        print('actual rating: %1.1f' % ratings[col])
    else:
        # if args.top = n, argpartition ensures that the top n elements are
        # on the far right of the array (sparing us from sorting the whole thing)
        partial_sort = np.argpartition(output, -args.top)
        top_n = partial_sort[-args.top:].tolist()
        top_n.sort(key=output.__getitem__)  # sort by rating
        print("The model predicts that the top %d movies for user %s will be: "
              % (args.top, args.user_id))
        for n in top_n:
            print
            print('%s: %1.2f' % (dat.column_to_name[n], output[n]))
            print('actual rating: %1.1f' % ratings[n])
