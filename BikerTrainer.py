from numpy.core.records import fromstring
import constants
from DemandPredictor import DemandPredictor
predictor = DemandPredictor()
import numpy as np

class Decision:
    def __init__(self, stationFrom, stationTo, demand):
        self.stationFrom = stationFrom
        self.stationTo = stationTo
        self.demand = demand

class BikerTrainer:

    # this should initilize everything related to storing previous values of decisions/policies etc.
    def __init__(self, env):
        self.env = env

    # This functions should update the reward associated with the state decision combination
    # Note: this implementation is currently myopic, so this is not required.
    # Simple Q-learning is implemented here by remembering which decision state pair had a good performance in the feature
    # optimal solutions require exact state and decision representations (impossible for this assignmnet)

    def update(self, state, decision, reward):
        return

    # This is where the decision happens; now a simple myopic heuristic rule is implemented. This does not consider future demand,
    # nor does it consider any intelligent vehicle routing/assignment
    # decisions or inventory replenishment decisions.

    # For the moment, the vehicle averages the current capacity of the current
    # stations with smallest and heighest capacity.
    def getDecision(self, state):

        # if bike is on the way, do nothing
        if state.arrivalTime > state.time:
            return Decision(-1, -1, -1)

        # the vehicle is available.
        return self.heuristicInventoryFillRate(state)
        

    def heuristicAveraging(self,state):
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

        return Decision(maxStation, minStation, fromMax) 
    
    def heuristicInventoryFillRate(self,state,fill_rate=0.5):
        expected_capacities = np.array(state.capacities) - predictor.predict_demand(self.env)
        expected_capacities_gap = (expected_capacities-constants.STATIONCAPACITY_PARAMETER*fill_rate)
        fromStation = expected_capacities_gap.argmax()
        toStation = expected_capacities_gap.argmin()
        avg = int(0.5 * (state.capacities[toStation] + state.capacities[fromStation]))

        fromMax = max(0,
                      min(state.vehicleCapAvail,
                          min(state.capacities[fromStation] - avg, avg - state.capacities[toStation])))
        return Decision(fromStation,toStation,fromMax)



