# HangmanKeras
Simple AI Agent Trained to play Hangman

This repo is a task for a coding interview. The tasks is to design an algorithm that is able to play hangman. They provided lists of words that can be used for training which unfortunately I am unable to disclose.
The algorithm can only be trained with the said list of words only. The algorithm then will be tested with out of vocabulary(that is given to me) words using an api. 
By fairness assumption, the test set should be similarly distributed (in case statistical approach is considered).

## Requirements
* Keras
* Tensorflow
* Numpy
* tqdm

## Approach Details
There are two approaches that comes to my mind for solving this problem. One is using reinforcement learning or casting the problem as supervised learning.
Since by the time I am working on this project I am not that familiar with reinforcement learning I decided to use the second approach.
During inference, the model I designed will output probability of letters existing on the word for a given state of the game and greedily select based on this probability.

## Model Description
I used neural network for my model. The neural network will receive two input, i.e. the guessed letters state which is a binarized state of the guessed letters and longest word length-padded state of the gameboard.
For the game board input I apply embeddings followed by two layers of LSTM and then with global average pooling
For the guessed letters state I used two stacks of dense layer.
Output of these two then combined and stacked with a layer of NN with softmax activation to generate probabilities.

## Model Training
The model collects the data by interacting directly with the environment.
During each global episode, the player will play a few batches of game using current state of the model and then use the words that should have been guessed as target.
Training is somehow faster with cpu due to simplicity of the model and possibly due to computational overhead with gpu.

## Things to improve
* Needs better parallelization
