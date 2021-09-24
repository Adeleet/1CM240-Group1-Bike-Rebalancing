import constants

class Decision:
    def __init__(self, stationFrom, stationTo, demand):
        self.stationFrom = stationFrom
        self.stationTo = stationTo
        self.demand = demand

class BikerTrainer:

    # this should initilize everything related to storing previous values of decisions/policies etc.
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

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

