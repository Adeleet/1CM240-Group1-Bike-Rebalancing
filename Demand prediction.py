import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import keras.layers

data = pd.read_csv('TestData.csv')
data['starttime'] = pd.to_datetime(data['starttime'], dayfirst=True)
data['stoptime'] = pd.to_datetime(data['stoptime'], dayfirst=True)

data['start_day'] = data['starttime'].dt.day
data.groupby(['start_day', 'end station id']).size().unstack().fillna(0)

data['starttime'] = data['starttime'].dt.round('1H')

data = data[['starttime', 'start station id']]
data = data.groupby(['starttime', 'start station id']).size().reset_index(name='count')


data['start_year'] = data['starttime'].dt.year
data['start_month'] = data['starttime'].dt.month
data['start_day'] = data['starttime'].dt.day
data['start_hour'] = data['starttime'].dt.hour
data['start_minute'] = data['starttime'].dt.minute
data.drop('starttime', axis=1, inplace=True)

data = pd.get_dummies(data, columns=['start station id'], drop_first=True)
first_column = data.pop('count')
data.insert(0, 'count', first_column)


from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
hyper_params = {}

# Initializing the regressors
xgb_clf = XGBRegressor()

# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):
    # Print iteration number to keep track of progress
    print("")
    print("Iteration: ", i + 1)
    print("")

    # kf = KFold(n_splits=5)
    train, test = train_test_split(data, test_size=0.2)
    x_train = train.iloc[:, 1:]
    y_train = train['count']

    x_test = test.iloc[:, 1:]
    y_test = test[['count']]
    # Training the models
    xgb_clf.fit(x_train, y_train)
    # Using the models to predict
    xgb["pred"] = xgb_clf.predict(x_test)
    # print('predict')
    # print(xgb_clf.predict(x_test.iloc[[25]]))
    # print('test')
    # print(y_test.iloc[[25]])
    # ghnn

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"]+' model r2 score: {0:0.4f}'. format(np.mean(modelType["r2"])))
    print(modelType["name"]+' model meanAE score: {0:0.4f}'. format(np.mean(modelType["meanAE"])))
    print(modelType["name"]+' model medianAE score: {0:0.4f}'. format(np.mean(modelType["medianAE"])))
    print(modelType["name"]+' model MAPE score: {0:0.4f}'. format(np.mean(modelType["MAPE"])))

