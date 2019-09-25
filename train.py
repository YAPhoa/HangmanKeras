import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tqdm import tqdm

from env import Hangman
from network import Network, NNAgent
from utils import *
from keras import optimizers

import numpy as np

with open('words.txt') as f :
    word_src = f.read().splitlines()

len_list = list(map(len, word_src))
maxlen = max(len_list)
print('Max length of words is', maxlen) 

policy_net = Network(maxlen=maxlen)
player = NNAgent(policy_net)
policy_net.summary()


save_episode = 5000
view_episode = 500
update_episode = 5
avg_correct = 0
wins_avg = 0
n_trials = 20000
progbar = tqdm(range(n_trials))

""" 
Warmup Phase
Since the network can only collect data by interacting with the environment, 
to collect more data max_lives parameter or the maximum number of trial before game over is set 
slightly extra than the actual game parameter so episodes are longer. 
"""

game_params = {'max_lives' : 8}
env = Hangman(word_src, **game_params)

print('Training Start ...', end = '\n\n')

for episode_set in progbar :
    for _ in range(update_episode) :
        state = env.reset()
        done = False
        correct_count = 0
        while not done :
            guess = player.select_action(state)
            state, reward, done, ans = env.step(guess)
            if reward > 0 :
                correct_count += 1.0
            if reward == env.win_reward :
                wins_avg += 1.0
        player.finalize_episode(ans['ans'])
        avg_correct += correct_count
    loss = player.train_model()
    progbar.set_description("Loss : {:.3f}              ".format(loss))

    if (episode_set +1) % view_episode == 0 :
        views = (episode_set + 1,avg_correct/(update_episode*view_episode), view_episode*update_episode, wins_avg/(update_episode*view_episode))
        print('Episode {} -------- Average Correct Count : {:.3f}     Last {} winrate : {:.3f}'.format(*views))
        if loss is not None :
            print('Loss :', loss)
            print()
            avg_correct = 0
            wins_avg = 0

    if (episode_set +1) % save_episode == 0 :
        player.model.save('policy.h5', include_optimizer=False)

print()
game_params = {'max_lives' : 6}
env = Hangman(word_src, **game_params)

""" 
Final Phase, 
Collect data by interacting with similar parameter as the actual game.
"""

n_trials = 70000
progbar = tqdm(range(n_trials))

player.model.compile(optimizer= optimizers.Adam(3e-5, clipnorm=1))

for episode_set in progbar :
    for _ in range(update_episode) :
        total_reward = 0    
        state = env.reset()
        done = False
        correct_count = 0
        while not done :
            guess = player.select_action(state)
            state, reward, done, ans = env.step(guess)
            #player.append_sample(letter_dict[guess] - 1, reward)
            if reward > 0 :
                correct_count += 1.0
            if reward == env.win_reward :
                wins_avg += 1.0
        player.finalize_episode(ans['ans'])
        avg_correct += correct_count
    loss = player.train_model()
    progbar.set_description("Loss : {:.3f}              ".format(loss))

    if (episode_set +1) % view_episode == 0 :
        views = (episode_set + 1,avg_correct/(update_episode*view_episode), view_episode, wins_avg/(update_episode*view_episode))
        print('Episode {} -------- Average Correct Count : {:.3f}     Last {} winrate : {:.3f}'.format(*views))
        if loss is not None :
            print('Loss :', loss)
            print()
            avg_correct = 0
            wins_avg = 0

    if (episode_set +1) % save_episode == 0 :
        player.model.save('policy.h5', include_optimizer=False)


