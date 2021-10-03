#%%
import numpy as np
import pandas as pd
import csv
import math
import copy
import gzip
from time import time
from copy import deepcopy

from BikerEnv import BikerEnv, Vehicle
from BikerTrainer import BikerTrainer
from constants import NITERATIONS_PARAMETER, OUTPUT_FLAG

#%%
env = BikerEnv("Test",use_sample=False)
#%%
trainer = BikerTrainer(env)


#%%
numberEpisodes = NITERATIONS_PARAMETER
obj = 0
for i in range(0, numberEpisodes):
    step = 1
    start_time = time()
    while (not env.game_over):

        state = env.getState()
        decision = trainer.getDecision(state)

        reward = env.step(decision)

        trainer.update(state, decision, reward)
        if OUTPUT_FLAG:
            print("END ITERATION at time: " + str(env.time))
        step += 1

    print("Objective of episode = " + str(env.objective)
          + "(" + str(env.acceptedBikes) + "/" + str(env.acceptedBikes + env.rejectedBikes)
          + " = "
          + str(int(float(env.acceptedBikes) / float(env.acceptedBikes + env.rejectedBikes) * 100.00)) + "%)")

    print(f"Took {step} steps in {time()-start_time:.1f} seconds")
    obj += env.objective

    env.reset()

print("Total perfomance = " + str(obj / NITERATIONS_PARAMETER))


#stations = env.readStationList()
# %%
