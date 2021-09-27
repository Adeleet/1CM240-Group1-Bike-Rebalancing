from numpy.core.fromnumeric import argmax
from tensorflow.python.ops.gen_array_ops import const
import constants
import numpy as np
from random import random, randint, sample
from copy import copy
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from VFA import ValueFunctionApproximator

class Decision:
    def __init__(self, stationFrom, stationTo, demand):
        self.stationFrom = stationFrom
        self.stationTo = stationTo
        self.demand = demand

    def as_array(self):
        return [self.stationFrom, self.stationTo, self.demand]

class BikerTrainer:

    # this should initilize everything related to storing previous values of decisions/policies etc.
    def __init__(self, env, replay_buffer_size=constants.REPLAY_BUFFER_SIZE):
        """
        Initialize (DQN) biker agent. The agent observes as State with time and vehicle capacity, and subsequentially takes a decision to (optionally) move a bike from one station to another.
        """
        self.env = env

        self.model = self.create_model()

    
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.target_update_count = 0

        self.VFA = ValueFunctionApproximator()

    def add_to_replay_buffer(self,state,decision,reward,new_state,done):
        """
        Adds transition (state, action, reward, new_state, done) to model replay memory buffer
        """
        self.replay_buffer.append([state,decision,reward,new_state,done])


    def create_model(self):
        """
        Returns a sequential neural network (Keras) with the states as input (334,) and outputs a decision to be made (3,).
        Input layer (334,): State.capacities and State.vehicleCapAvail
        Output layer (3,): Decision.stationFrom, Decision.stationTo, Decision.demand
        """
        numStations = 333
        demandRange = 20
        output_size = numStations*numStations*demandRange
        model = Sequential([
            Dense(300,activation='relu',input_shape=(334,)),
            Dense(300,activation='relu'),
            Dense(output_size),
        ],name='BikerTrainer')
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def train(self,game_over,step):
        """
        Train sequential neural network if possible (min. replay buffer reached) and update Q_values

        Training consists of sampling from the replay buffer (observed transitions) and using the NN to predict decisions/Q-values
        
        The main model is used to make predictions for the current states, whereas the target model is used to make predictions for the 'next_states'
        """
        if len(self.replay_buffer) < constants.MIN_REPLAY_BUFFER_SIZE:
            return
        
        batch = sample(self.replay_buffer, constants.BATCH_SIZE)

        current_states = np.array([transition[0].as_array() for transition in batch])

        # TODO use VFA here instead of lookup table

        current_Q_values = self.model.predict(current_states).reshape(-1,333,333,20)

        future_states = np.array([transition[3].as_array() for transition in batch])
        future_Q_values = self.target_model.predict(future_states).reshape(-1,333,333,20)


        X = []
        y = []
    
        for i, (current_state, decision, reward, new_state, done) in enumerate(batch):

            if not done:
                max_reward = future_Q_values[i].max()
                updated_Q = reward + constants.DISCOUNT * max_reward
            else: 
                updated_Q = reward
            
    
            current_Q = current_Q_values[i].reshape(333,333,20)
            current_Q[decision.stationFrom,decision.stationTo,decision.demand] = updated_Q

            X.append(current_state.as_array())
            y.append(current_Q)
        
        self.model.fit(np.array(X),np.array(y).reshape(constants.BATCH_SIZE,-1),batch_size=constants.BATCH_SIZE, shuffle=False)

        # Update target network after game over
        if game_over:
            self.target_update_count += 1
        
        # Update target network with weights of main network if counter exceeds threshold 
        if self.target_update_count > constants.STEPS_BETWEEN_UPDATE:
            self.target_model.set_weights(self.model_get_weights())
            self.target_update_count = 0


    def predict(self, states):
        predictions = super().predict(states)
        n = predictions.reshape(-1,333,333,20).shape[0]
        argmax_probs = predictions.reshape(n,-1).argmax(axis=-1)
        return np.unravel_index(argmax_probs,(333,333,20))
            


    # This functions should update the reward associated with the state decision combination
    # Note: this implementation is currently myopic, so this is not required.
    # Simple Q-learning is implemented here by remembering which decision state pair had a good performance in the feature
    # optimal solutions require exact state and decision representations (impossible for this assignmnet)

    def update(self, state, decision, reward):
        state_action_pairs = np.array(self.replay_buffer)[:,:-1]
        rewards = np.array(self.replay_buffer)[:,-1]

        self.model.fit(state_action_pairs,rewards,epochs=5,verbose=0)
        return

    def get_Q_values(self,state):
        return self.model.predict(state.as_array())

    def replay(self,state):
        replay_env = copy(self.env)

        for i in range(100):
            print(f"Episode {i+1}")
            # Initiate 100 runs, starting from current environment state
            episode_env = copy(replay_env)
            episode_state = episode_env.getState()

            while not episode_env.game_over:
                
                if random()>0.5:
                    decision = self.greedy_action(episode_state)
                else:
                    decision = self.random_action(episode_state)

                current_state = episode_state.as_array()

                reward = episode_env.step(decision)

                self.update([current_state,decision,reward])

        return self.replay_buffer
            # return np.array(Q_values).argmax(axis=3)
            




    # This is where the decision happens; now a simple myopic heuristic rule is implemented. This does not consider future demand,
    # nor does it consider any intelligent vehicle routing/assignment
    # decisions or inventory replenishment decisions.

    

    # For the moment, the vehicle averages the current capacity of the current
    # stations with smallest and heighest capacity.
    def getDecision(self, state):   
        # if bike is on the way, do nothing
        if state.arrivalTime > state.time:
            return Decision(-1, -1, -1)
        

        self.replay(state)
        
        state,decision,reward = np.array(self.replay_buffer).argmin(axis=3)
        
        
    def greedy_action(self,state):
        q_values = self.get_Q_values(state)
        return np.argmin(q_values)

    def heuristic_action(self,state):

        # the vehicle is available.
        # find station with minimum and maximum number of capacity.
        minStation = state.capacities.index(min(state.capacities))
        maxStation = state.capacities.index(max(state.capacities))

        if abs(min(state.capacities) - max(state.capacities)) <= 2:
            return Decision(-1, -1, -1)

        # determine how many bikes to take from max to min station
        avg = int(0.5 * (state.capacities[minStation] + state.capacities[maxStation]))

        fromMax = max(0,
                      min(state.vehicleCapAvail,
                          min(state.capacities[maxStation] - avg, avg - state.capacities[minStation])))

        # return the decision from minstation, to maxstation, frommax

        if constants.OUTPUT_FLAG:
            print("Decision is taken at time: " + str(state.time))
            print("from station " + str(maxStation) + " of current cap: " + str(max(state.capacities)))
            print("to   station " + str(minStation) + " of current cap: " + str(min(state.capacities)))
            print("Quantity: " + str(fromMax))
            print()

        return Decision(maxStation, minStation, fromMax)

    def random_action(self,state):
        if len(state.capacities)<=1:
            raise RuntimeError(f"State capacities length: {state.capacities}\nState: {state.as_array()}")
        maxStation = randint(0,len(state.capacities)-1)
        minStation = randint(0,len(state.capacities)-1)

        max_demand = min(state.capacities[maxStation],20)
        
        demand = randint(0,max_demand)
        
        return Decision(maxStation, minStation, demand)