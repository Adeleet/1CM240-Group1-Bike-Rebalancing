# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:32:14 2021

@author: Albert Schrotenboer

Note, we need the TestData in the same folder to be able to run the code!
"""
#%%

from random import random
import numpy as np
import pandas as pd
import csv
import math
import copy
import gzip
from tqdm import tqdm


from BikerEnv import BikerEnv
from BikerTrainer import BikerTrainer
from constants import EPSILON_DECAY, MIN_EPSILON, NITERATIONS_PARAMETER, OUTPUT_FLAG, STEPS_BETWEEN_UPDATE

#%%
env = BikerEnv("Test")
trainer = BikerTrainer(env)

epsilon = 1  # not a constant, going to be decayed


numberEpisodes = NITERATIONS_PARAMETER
obj = 0
steps_since_update = 0 
episode_rewards = []

#%%
for i in tqdm(range(0, numberEpisodes)):
    episode_reward = 0
    current_state = env.reset()
    done = False
    step = 1

    while not done:

        if random() > epsilon:
            # Take greedy action using Q values
            decision = trainer.greedy_action(current_state)
        else:
            # Take random action
            decision = trainer.random_action(current_state)
        
        # Send action to environment and observe new state, reward and done (game over)
        new_state, reward, done = env.step(decision)

        episode_reward += reward

        trainer.add_to_replay_buffer(current_state,decision,reward,new_state,done)
        trainer.train(done,step)

        current_state = new_state
        step += 1

    episode_rewards.append(episode_reward)

    # Decay epsilon parameter
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, MIN_EPSILON)
    
    env.print_episode()


# %%
