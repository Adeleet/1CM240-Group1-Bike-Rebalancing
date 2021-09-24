# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:32:14 2021

@author: Albert Schrotenboer

Note, we need the TestData in the same folder to be able to run the code!
"""


import numpy as np
import pandas as pd
import csv
import math
import copy
import gzip


from BikerEnv import BikerEnv
from BikerTrainer import BikerTrainer
import constants


env = BikerEnv("Test")
trainer = BikerTrainer(1, 2)


numberEpisodes = NITERATIONS_PARAMETER
obj = 0
for i in range(0, numberEpisodes):

    # print()
    # print()
    # for stat in env.stations:
    #    print(stat.currentCap, end = ' ')
    # print()

    while (not env.game_over):

        if OUTPUT_FLAG:
            print()
            print()
            print()
            print("NEW ITERATION at time: " + str(env.time))

        state = env.getState()
        decision = trainer.getDecision(state)

        reward = env.step(decision)

        trainer.update(state, decision, reward)
        if OUTPUT_FLAG:
            print("END ITERATION at time: " + str(env.time))

    # for stat in env.stations:
    #    print(stat.currentCap, end = ' ')

    # print()
    print("Objective of episode = " + str(env.objective)
          + "(" + str(env.acceptedBikes) + "/" + str(env.acceptedBikes + env.rejectedBikes)
          + " = "
          + str(int(float(env.acceptedBikes) / float(env.acceptedBikes + env.rejectedBikes) * 100.00)) + "%)")

    obj += env.objective

    env.reset()

print("Total perfomance = " + str(obj / NITERATIONS_PARAMETER))


#stations = env.readStationList()