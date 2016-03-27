import argparse

import subprocess
import model

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train model from scratch')
    parser.add_argument('-u', '--user_id', type=int, help='user to predict ratings for')
    parser.add_argument('-m', '--movie_id', type=int, help='movie to predict ratings for')
    args = parser.parse_args()

    # check if user has entered both a username and a movie title
    username_and_movie_warning = 'Warning: the model cannot make predictions ' \
                                 'unless you enter both a username and a movie title.'
    if args.user_id and not args.movie_id:
        print(username_and_movie_warning)
        args.movie_id = input('Please enter the movie you would like '
                           'to predict %s\'s ratings for: ' % args.user_id)
    if args.movie_id and not args.user_id:
        print(username_and_movie_warning)
        args.username = input('Please enter the user whose rating of %s '
                              'you would like to predict: ' % args.movie_id)

    # path to saved version of trained model
    load_path = os.path.join('checkpoints', 'checkpoint')

    # check if a model has been previously trained
    already_trained = os.path.exists(load_path)
    if not (args.train or already_trained):
        response = input('Model has not been trained. '
                         'Train it now (this may take several hours)?')
        while True:
            if response in 'Yes yes':
                args.train = True
                break
            elif response in 'No no':
                print('Ok. Goodbye.')
                exit(0)
            else:
                response = input('Please enter [y|n].')

    if args.train:
        subprocess.call(['python', 'model.py'])

    # predict a rating for the user
    if args.user_id and args.movie_id:
        prediction = model.predict(args.user_id, args.movie_id)
        print("The model predicts that %s will give %s a %d"
              % args.user_id, args.movie_id, prediction)
