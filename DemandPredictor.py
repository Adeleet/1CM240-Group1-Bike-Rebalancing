import joblib
import numpy as np


class DemandPredictor:
    def __init__(self):
        self.xgboostmodel = joblib.load('xgboost.joblib')

    def predict_demand(self, env):
        """

        :param env: the environment, the station long and latitudes and hour will be determined from this
        :return: predicted demand in integer
        """
        X = np.zeros((len(env.stations), 3))
        for e, station in enumerate(env.stations):
            long = station.longitude
            lat = station.latitute
            hour = env.hour
            X[e][0] = lat
            X[e][1] = long
            X[e][2] = hour + 1

        return self.xgboostmodel.predict(X).clip(min=0).round().astype('int')



