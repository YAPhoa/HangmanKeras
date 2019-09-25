import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from keras.layers import LSTM, Embedding, Input, Dense, GlobalAveragePooling1D, Concatenate,Bidirectional
from keras.models import Model
from keras import optimizers, backend as K

from warnings import filterwarnings

filterwarnings('ignore')
import random
from utils import letter_dict, letters, pad_sequences

class Network(object) :
    """Define the network
    network consists of two input that reads the current state 
    and one hot encoded matrix of guessed letters
    This class includes some helper function to ease training and inference
    """
    def __init__(self, maxlen = 29) :
        state_embedding = self.get_state_embedding(maxlen)
        guessed_embedding = self.get_guessed_embedding()
        x = Concatenate()([state_embedding.output, guessed_embedding.output])
        x = Dense(100, activation = 'tanh')(x)
        x = Dense(26, activation = 'softmax')(x)
        self.full_model = Model([state_embedding.input, guessed_embedding.input], x, name = 'fullmodel')
        self.compile()
        
    def get_state_embedding(self, maxlen = 29) :
        inp = Input(shape = (maxlen,))
        x = Embedding(30,100, mask_zero = True)(inp)
        x = Bidirectional(LSTM(100 , dropout = 0.2, return_sequences=True))(x)
        x = Bidirectional(LSTM(100, dropout = 0.2 , return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(100, activation = 'tanh')(x)
        return Model(inp,x , name = 'StateEmbedding')
    
    def get_guessed_embedding(self) :
        inp = Input(shape = (26,))
        x = Dense(60, activation = 'tanh')(inp)
        x = Dense(60, activation = 'tanh')(x)
        return Model(inp, x, name = 'GuessedEmbedding')

    def __call__(self, state, guessed) :
        return self.full_model.predict([state,guessed]).flatten()
    
    def fit(self, *args, **kwargs) :
        return self.full_model.fit(*args, **kwargs)
    
    def train_on_batch(self, *args, **kwargs) :
        return self.full_model.train_on_batch(*args, **kwargs)        
    
    def summary(self) :
        self.full_model.summary()

    def save(self, *args, **kwargs) :
        self.full_model.save(*args, **kwargs)

    def load_weights(self, *args, **kwargs) :
        self.full_model.load_weights(*args, **kwargs)
        self.compile()

    def compile(self, optimizer= None) : 
        if optimizer is not None :
            self.full_model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
        else :
            self.full_model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(1e-3, clipnorm=1))

class Agent(object) :

    """Agent definition.
    Agent is embedded with a model and policy
    Agent can use stochastic policy i.e. choose action randomly from computed probability 
    or greedy i.e choose the most probable action out of unused actions.
    
    Agent is trained off-policy, after a set amount of episode (in my case I trained with 3 episodes), 
    and after each episode during training must be finalized with finalize_episode method 
    to compute the correct course of actions.

    train_model method will collect accumulated episodes and perform one iteration of gradient descent
    with the collected episode data.

    Tried both with stochastic and greedy. Greedy policy performs converges better.
    """

    def __init__(self, model, policy = 'greedy') :
        self.reset_guessed()
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = 'greedy'
        self.policy = property(self.get_policy, self.set_policy)
        self.reset_guessed()
        self.is_training = True
        self.model = model

    @staticmethod
    def guessed_mat(guessed) :
        mat = np.empty([1,26])
        for i, l in enumerate(letters) :
            mat[0,i] = 1 if l in guessed else 0
        return mat
    
    def get_guessed_mat(self) :
        return self.guessed_mat(self.guessed)

    def reset_guessed(self) :
        self.guessed = []

    def get_probs(self, state) :
        raise NotImplementedError()

    def get_policy(self) :
        return self._policy

    def set_policy(self, policy) :
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = policy

    def select_action(self,state) :            
        probs = self.get_probs(state)
        if self._policy == 'greedy' :
            i = 1
            sorted_probs = probs.argsort()
            while letters[sorted_probs[-i]] in self.guessed :
                i+= 1
            idx_act = sorted_probs[-i]
        elif self._policy == 'stochastic' :
            idx_act = np.random.choice(np.arange(probs.shape[0]), p = probs)
        guess = letters[idx_act]
        if guess not in self.guessed :
            self.guessed.append(guess)
            
        return guess
        
    def eval(self) :
        self.is_training = False
        self.set_policy('greedy') 
    
    def train(self) :
        self.is_training = True

class NNAgent(Agent) :
    def __init__(self, model, maxlen=29, policy='greedy') :
        super().__init__(model, policy)
        self.episode_memory = []
        self.states_history = []
        self.maxlen = maxlen

    def train_model(self):
        inp_1, inp_2, obj = zip(*self.states_history)
        inp_1 = np.vstack(list(inp_1)).astype(float)
        inp_2 = np.vstack(list(inp_2)).astype(float)
        obj = np.vstack(list(obj)).astype(float)
        loss = self.model.train_on_batch([inp_1,inp_2], obj)
        self.states_history = []
        return loss

    def get_probs(self, state) :
        state = self.preprocess_input(state)
        probs = self.model(*state)
        probs /= probs.sum()
        return probs

    def finalize_episode(self, answer) :
        inp_1, inp_2 = zip(*self.episode_memory)
        inp_1 = np.vstack(list(inp_1)).astype(float)      #stack the game state matrix
        inp_2 = np.vstack(list(inp_2)).astype(float)      #stack the one hot-encoded guessed matrix
        obj = 1.0 - inp_2                                 #compute the unused letters one-hot encoded
        len_ep = len(self.episode_memory)                 #length of episode
        correct_mask = np.array([[1 if l in answer else 0 for l in letters]]) # get mask from correct answer
        correct_mask = np.repeat(correct_mask, len_ep, axis = 0).astype(float)
        obj = correct_mask * obj  #the correct action is choosing the letters that are both unused AND exist in the word
        obj /= obj.sum(axis = 1).reshape(-1,1) #normalize so it sums to one
        self.states_history.append((inp_1, inp_2,obj))
        self.episode_memory = []
        self.reset_guessed()
    
    def preprocess_input(self, state) :
        new_input = []
        for l in state :
            new_input.append(letter_dict[l])
        state = pad_sequences([new_input], maxlen = self.maxlen)
        if self.is_training :
            self.episode_memory.append((state,self.get_guessed_mat()))
        return state, self.get_guessed_mat()



if __name__ =='__main__' :
    pass