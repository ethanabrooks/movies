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
Full disclosure. This is currently a work in progress. Tensorflow is hard.
As you can see in the code Movies and Books are classes that inherit from the
main class, data, and override the methods involved in parsing data.

The main optimization is the use of a randomized embeddings matrix instead of a
a sparse vector representing every single movie/book. The goal was to avoid
dealing with this large sparse matrix. Because the vectors in a randomized
embeddings matrix are, on average, orthogonal, when we add these vectors, we
achieve an effect similar to a large sparse matrix with ones where there is data
and zeros where there is none. Furthermore, if we multiply each of these
randomized embeddings vectors by the rating, we achieve a similar effect to a
large sparse matrix with zeros where there is no data and the rating where there
is data. The major benefit is that we avoid filling thousands of memory
addresses with zeros, as we would have to with the sparse matrix.
