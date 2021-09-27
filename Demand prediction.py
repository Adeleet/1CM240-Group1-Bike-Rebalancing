import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

#%%
# Import dataset
data = pd.read_csv('TestData.csv')

# Set dates to datetime objects
data['starttime'] = pd.to_datetime(data['starttime'], dayfirst=True)
data['stoptime'] = pd.to_datetime(data['stoptime'], dayfirst=True)

# Add weekday column to dataset
data['weekday'] = data['starttime'].dt.dayofweek
data['starttime'] = data['starttime'].dt.round('1H')
# Seperate data to smaller dataset only including weekday and starttime
data = data[['starttime', 'weekday', 'start station latitude', 'start station longitude']]
# Count demand in 1 hour timeframe by grouping on same rows. Counted demand per hour on weekday
data = data.groupby(['starttime', 'weekday', 'start station latitude', 'start station longitude']).size().reset_index(name='count')

# it is chosen to ontly take hour and weekday into account as we only have data for 1 month of a single year
data['start_hour'] = data['starttime'].dt.hour
data.drop('starttime', axis=1, inplace=True)
first_column = data.pop('count')
data.insert(0, 'count', first_column)

#%%
# import clustering function and predict cluster
KMeans = joblib.load('KMeans_Function.joblib')

y_pred = KMeans.predict(data)
pred = pd.DataFrame(y_pred)
pred.columns = ['ClusterType'] #store K-means results in a dataframe
X = pd.DataFrame(data, columns=['count', 'weekday', 'start station latitude', 'start station longitude', 'start_hour'])
dataClusterd = pd.concat([data, pred], axis=1)

# Create new datasets based on clusters
dataCluster1 = dataClusterd[dataClusterd['ClusterType'] == 0]
dataCluster2 = dataClusterd[dataClusterd['ClusterType'] == 1]
dataCluster3 = dataClusterd[dataClusterd['ClusterType'] == 2]
dataCluster4 = dataClusterd[dataClusterd['ClusterType'] == 3]
dataCluster5 = dataClusterd[dataClusterd['ClusterType'] == 4]

data_list = [dataCluster1, dataCluster2, dataCluster3, dataCluster4, dataCluster5]

#%%
# Hyper parameter optimization by grid search Values taken arbitrarily
models = {}
params = {'max_depth': [3, 6, 10],
          'learning_rate': [0.01, 0.05, 0.1],
          'n_estimators': [100, 500, 1000],
          'colsample_bytree': [0.3, 0.7]}

regressors = []
for i in range(0, 5):
    regressors.append(XGBRegressor())

for e, d in enumerate(data_list):
    print('model ' + str(e))
    # Initializing lists with performance metrics
    xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
    for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
        xgb[performanceMetric] = []

    # split dataset
    train, test = train_test_split(d, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models and doing grid search for hyperparameter optimization
    regressors[e].fit(x_train, y_train)
    clf = GridSearchCV(cv=3, estimator=regressors[e], param_grid=params, scoring='neg_mean_squared_error', verbose=1)
    clf.fit(x_train, y_train)
    print('best parameters model'+ str(e), ':', clf.best_params_)
    print('Lowest RMSE: ', (-clf.best_score_)**(1/2.0))


#%%
# Train model for cluster 1
XGBoost1 = XGBRegressor(colsample_bytree=0.7, learning_rate=0.1, max_depth=6, n_estimators=500)

print('model 1')
# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):

    # split dataset
    train, test = train_test_split(dataCluster1, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models
    XGBoost1.fit(x_train, y_train)

    # Using the models to predict
    xgb["pred"] = XGBoost1.predict(x_test)

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))
        modelType['RMSE'].append(mean_squared_error(y_test, modelType['pred']))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"]+' model r2 score: {0:0.4f}'. format(np.mean(modelType["r2"])))
    print(modelType["name"]+' model meanAE score: {0:0.4f}'. format(np.mean(modelType["meanAE"])))
    print(modelType["name"]+' model medianAE score: {0:0.4f}'. format(np.mean(modelType["medianAE"])))
    print(modelType["name"]+' model MAPE score: {0:0.4f}'. format(np.mean(modelType["MAPE"])))
    print(modelType["name"] + ' model RMSE score: {0:0.4f}'.format(np.mean(modelType["RMSE"])))

joblib.dump(XGBoost1, 'xgboostCluster1.joblib')

#%%
# Train model for cluster 2
XGBoost2 = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, n_estimators=500)

print('model 2')
# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):
    # split dataset
    train, test = train_test_split(dataCluster2, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models
    XGBoost2.fit(x_train, y_train)

    # Using the models to predict
    xgb["pred"] = XGBoost2.predict(x_test)

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))
        modelType['RMSE'].append(mean_squared_error(y_test, modelType['pred']))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"] + ' model r2 score: {0:0.4f}'.format(np.mean(modelType["r2"])))
    print(modelType["name"] + ' model meanAE score: {0:0.4f}'.format(np.mean(modelType["meanAE"])))
    print(modelType["name"] + ' model medianAE score: {0:0.4f}'.format(np.mean(modelType["medianAE"])))
    print(modelType["name"] + ' model MAPE score: {0:0.4f}'.format(np.mean(modelType["MAPE"])))
    print(modelType["name"] + ' model RMSE score: {0:0.4f}'.format(np.mean(modelType["RMSE"])))

joblib.dump(XGBoost2, 'xgboostCluster2.joblib')
#%%
# Train model for cluster 3
XGBoost3 = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, n_estimators=500)

print('model 3')
# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):
    # split dataset
    train, test = train_test_split(dataCluster3, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models
    XGBoost3.fit(x_train, y_train)

    # Using the models to predict
    xgb["pred"] = XGBoost3.predict(x_test)

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))
        modelType['RMSE'].append(mean_squared_error(y_test, modelType['pred']))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"] + ' model r2 score: {0:0.4f}'.format(np.mean(modelType["r2"])))
    print(modelType["name"] + ' model meanAE score: {0:0.4f}'.format(np.mean(modelType["meanAE"])))
    print(modelType["name"] + ' model medianAE score: {0:0.4f}'.format(np.mean(modelType["medianAE"])))
    print(modelType["name"] + ' model MAPE score: {0:0.4f}'.format(np.mean(modelType["MAPE"])))
    print(modelType["name"] + ' model RMSE score: {0:0.4f}'.format(np.mean(modelType["RMSE"])))

joblib.dump(XGBoost3, 'xgboostCluster3.joblib')

#%%
# Train model for cluster 4
XGBoost4 = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, n_estimators=1000)

print('model 4')
# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):
    # split dataset
    train, test = train_test_split(dataCluster4, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models
    XGBoost4.fit(x_train, y_train)

    # Using the models to predict
    xgb["pred"] = XGBoost4.predict(x_test)

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))
        modelType['RMSE'].append(mean_squared_error(y_test, modelType['pred']))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"] + ' model r2 score: {0:0.4f}'.format(np.mean(modelType["r2"])))
    print(modelType["name"] + ' model meanAE score: {0:0.4f}'.format(np.mean(modelType["meanAE"])))
    print(modelType["name"] + ' model medianAE score: {0:0.4f}'.format(np.mean(modelType["medianAE"])))
    print(modelType["name"] + ' model MAPE score: {0:0.4f}'.format(np.mean(modelType["MAPE"])))
    print(modelType["name"] + ' model RMSE score: {0:0.4f}'.format(np.mean(modelType["RMSE"])))

joblib.dump(XGBoost4, 'xgboostCluster4.joblib')

#%%
# Train model for cluster 5
XGBoost5 = XGBRegressor(colsample_bytree=0.7, learning_rate=0.1, max_depth=6, n_estimators=100)

print('model 5')
# Initializing lists with performance metrics
xgb = {"name": "XGBoost", "confusion_matrix": np.zeros(shape=(3, 3))}
for performanceMetric in ["r2", "meanAE", "medianAE", "MAPE", "RMSE"]:
    xgb[performanceMetric] = []

# Perform cross-validation multiple times to get a better estimate
for i in range(12):
    # split dataset
    train, test = train_test_split(dataCluster5, test_size=0.2)
    x_train = train.iloc[:, 1:-1]
    y_train = train['count']
    x_test = test.iloc[:, 1:-1]
    y_test = test[['count']]

    # Training the models
    XGBoost5.fit(x_train, y_train)

    # Using the models to predict
    xgb["pred"] = XGBoost5.predict(x_test)

    # Saving stats
    for modelType in [xgb]:
        modelType["r2"].append(r2_score(y_test, modelType["pred"]))
        modelType["meanAE"].append(mean_absolute_error(y_test, modelType["pred"]))
        modelType["medianAE"].append(median_absolute_error(y_test, modelType["pred"]))
        modelType["MAPE"].append(mean_absolute_percentage_error(y_test, modelType["pred"]))
        modelType['RMSE'].append(mean_squared_error(y_test, modelType['pred']))

# Summarizing results
for modelType in [xgb]:
    print(modelType["name"] + ' model r2 score: {0:0.4f}'.format(np.mean(modelType["r2"])))
    print(modelType["name"] + ' model meanAE score: {0:0.4f}'.format(np.mean(modelType["meanAE"])))
    print(modelType["name"] + ' model medianAE score: {0:0.4f}'.format(np.mean(modelType["medianAE"])))
    print(modelType["name"] + ' model MAPE score: {0:0.4f}'.format(np.mean(modelType["MAPE"])))
    print(modelType["name"] + ' model RMSE score: {0:0.4f}'.format(np.mean(modelType["RMSE"])))

joblib.dump(XGBoost5, 'xgboostCluster5.joblib')
