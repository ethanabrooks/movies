import argparse

import subprocess

import data
import model

import os


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


if __name__ == '__main__':
    import main

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train model from scratch')
    parser.add_argument('-u', '--user_id', type=int, help='user to predict ratings for')
    parser.add_argument('-m', '--movie_id', type=int, help='movie to predict ratings for')
    parser.add_argument('-t', '--top', type=int, help='the top [n] highest rated movies'
                                                      'predicted for user')
    args = parser.parse_args()

    # check if user has entered both a username and a movie title
    if not args.movie_id and not args.top:
        args.movie_id = prompt_for_int('Please enter the movie you would like '
                                       'to predict %s\'s ratings for: ' % args.user_id)
    if not args.user_id:
        args.user_id = prompt_for_int('Please enter the user whose rating of %s '
                                      'you would like to predict: ' % args.movie_id)

    # path to saved version of trained model
    load_path = os.path.join('checkpoints', 'checkpoint')

    # check if a model has been previously trained
    already_trained = os.path.exists(load_path)
    if not (args.train or already_trained):
        main.check_if_ok_to_continue('Model has not been trained. '
                                     'Train it now (this may take several hours)? ')
    args.train = True

    if args.train:
        subprocess.call(['python', 'model.py'])

    # predict a rating for the user
    if args.user_id and args.movie_id:
        data_data = data.Data(ratings='debug.dat')
        data_data.get_ratings(2)
        prediction = model.predict(args.user_id, args.movie_id, data_data)
        print("The model predicts that %s will give %s a %d"
              % args.user_id, args.movie_id, prediction)
