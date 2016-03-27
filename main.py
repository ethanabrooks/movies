import argparse

import subprocess

import model

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train model from scratch')
    parser.add_argument('-u', '--userid', type=int, help='user to predict ratings for')
    parser.add_argument('-m', '--movieid', type=int, help='movie to predict ratings for')
    args = parser.parse_args()

    # check if user has entered both a username and a movie title
    username_and_movie_warning = 'Warning: the model cannot make predictions ' \
                                 'unless you enter both a username and a movie title.'
    if args.username and not args.movie:
        print(username_and_movie_warning)
        args.movie = input('Please enter the movie you would like '
                           'to predict %s\'s ratings for: ' % args.username)
    if args.movie and not args.username:
        print(username_and_movie_warning)
        args.username = input('Please enter the user whose rating of %s '
                              'you would like to predict: ' % args.movie)

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
    if args.userid and args.movie:
        prediction = model.predict(args.username, args.movie)
        print("The model predicts that %s will give %s a %d"
              % args.username, args.movie, prediction)
