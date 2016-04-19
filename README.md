First, this code depends on several packages. To install, simpy run:

    pip install -r requirements.txt

The model is designed to be used on the command line, using arguments to main.
Run

    python main.py -h
to get instructions on possible arguments, although if any necessary arguments
are missing, the program will prompt you for them.
 
The first time you run `main.py` the model will create the necessary file
structure and then quit. It is then up to the user to actually populate with
files (to lazy to code in `wget`). Afterward, `main.py` will actually run the
model.

In order to run each part of the model in isolation, it is advisable to follow
the these steps:

1. Run `python movies.py` (or the dataset of your choosing) to parse the data.

2. Run `model.py --dataset=Movies --retrain` to train the model. Of course, this
   can take a while.

3. Finally, 

The main goal of this design (besides just getting TensorFlow to work!) was to
allow the program's three main functions to operate independently:

1. interacting with the user / making predictions (handled by main.py)

2. loading data (handled by data.py)

3. training the model (handled by model.py and ops.py)

Changes to the functionality of the program should only affect main.py,
as long as they make use of the model's output of a list of predicted ratings
across the range of movies. (Most) changes to the dataset should only affect
data.py, and changes to the architecture do only affect model.py and ops.py.

Checkpoint 2:
The model implements 3 datasets:
- The original short movies dataset (`easy_movies.py`)
- The larger movies dataset (`movies.py`)
- THe books dataset (`books.py`)

Each of these modules implements a class that inherits from the abstact
`Data` object, which does most of the heavy lifting, while these smaller
modules just take care of parsing. Any dataset that implements just two simple
abstract methods can be plugged into the model.

The model has two main performance features: 
-Instead of loading all the training data into memory, the model pulls data from 
a file pointer that remains open during training. This actually slows down the model, but makes it capable of handling much larger datasets.
-The model can pause and resume training on the fly (it saves its parameters to
disk and automatically checks for them upon resuming training).
