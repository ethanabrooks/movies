First, this code depends on several packages. To install, simpy run:

    pip install -r requirements.txt

The model is designed to be used on the command line, using arguments to main.
Run

    python main.py -h
to get instructions on possible arguments, although if any necessary arguments
are missing, the program will prompt you for them.

The main goal of this design (besides just getting TensorFlow to work!) was to
allow the program's three main functions to operate independently:

1. interacting with the user / making predictions (handled by main.py)

2. loading data (handled by data.py)

3. training the model (handled by model.py and ops.py)

Changes to the functionality of the program should only affect main.py,
as long as they make use of the model's output of a list of predicted ratings
across the range of movies. (Most) changes to the dataset should only affect
data.py, and changes to the architecture do only affect model.py and ops.py.

Notable features:
- The model only has to preprocess data once. After that, it automatically
  detects the necessary files and loads metadata into memory. (To reload data, run ``python data.py reload``)
- The model continuously saves its parameters to memory so that it can pickup
  training where it left off.


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
- Instead of loading all the training data into memory, the model pulls data
  from a file pointer that remains open during training. This actually slows
down the model, but makes it capable of handling much larger datasets.
- On the inputs side, the model hashes into a randomized embeddings vector
  (rather than representing the user's ratings as a massive sparse vector).
Again, the performance savings are more on the side of memory here.
