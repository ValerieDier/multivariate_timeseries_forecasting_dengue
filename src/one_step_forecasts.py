from pandas import DataFrame, concat
from numpy import asarray
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from timeseries_data_prep import train_test_split_rows_reserved

def random_forest_forecast(train, testX):
	"""Fit a random forest model and make a one step prediction
	
	Reads in a training set, and if used as originally intended, one point
	from a test set.
	
	Note: a full test set can likely be sent, and it will generate
	predictions from this set, but will only return the first prediction.
	
	It takes the training data and divides it into the inputs (X)
	and output (y) and fit the regressor on these.
	
	It will then generate a prediction on the test data (testX) and return the 
	first prediction.  Only one prediction is generated if only one row
	of the test X data is sent to the function.
	
	Caution: 
	1. Proper functioning depends on the use of the series_to_supervised
	function(s).
	2. Do not send more than one output column (target), that is, y(t) to
	this function.  This is determined in the n_out argument sent to 
    series_to_supervised.
	
	For a univariate timeseries model, the inputs are the lagged
	target values and the output is the target value at time t presuming
	no target columns were generated in prior data processing for time t+k.
	If columns were generated for time t+k, the function will only use the last 
	column sent as the output and will put the other target columns in the 
	input (trainX) set along with the lagged inputs.  Additional processing
    of the results of series_to_supervised would be required prior to sending
    to walk_fwd_validation_rf.
	
	Parameters
	-----------
	train : training data
	testX : test data for the inputs or features
	
	With thanks to Jason Brownlee's Machine Learning Mastery site.
	"""

	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]


def walk_fwd_validation_rf(data, n_test):
    """Walk-forward validation for Random Forest regressor
	
	Reads in data and calls train_test_split_rows_reserved to generate
	a training set and a test set.
	
	The training set is stored as "history" for fitting the regressor.  
	It is sent to random_forest_forecast for this purpose.  One row of the
	test set is also sent to random_forest_forecast to generate a one-step
	prediction and calculate error.  This test row is then appended to
	"history" for the next iteration in fitting the regressor; it is now
	"seen data".
	
	Predictions and actual values are displayed on each iteration.  Error
	is displayed at the end of all iterations using the accumulated 
	predictions and "seen" test rows.
	
	This effectively performs a real-time refitting of the regressor as 
	new observations (test rows) come in. 
	
	Parameters
	-----------
	data : dataset prior to splitting into train and test sets
	n_test : number of rows desired for test set, taken from the end
             of the dataset

	With thanks to Jason Brownlee's Machine Learning Mastery site.
	"""

    predictions = list()
    train, test = train_test_split_rows_reserved(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = random_forest_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


def xgboost_forecast(train, testX):
	"""Fit an xgboost model and make a one step prediction
	
	Reads in a training set, and if used as originally intended, one point
	from a test set.
	
	Note: a full test set can likely be sent, and it will generate
	predictions from this set, but will only return the first prediction.
	
	It takes the training data and divides it into the inputs (X)
	and output (y) and fit the regressor on these.
	
	It will then generate a prediction on the test data (testX) and return the 
	first prediction.  Only one prediction is generated if only one row
	of the test X data is sent to the function.
	
	Caution: 
	1. Proper functioning depends on the use of the series_to_supervised
	function(s).
	2. Do not send more than one output column (target), that is, y(t) to
	this function.  This is determined in the n_out argument sent to 
    series_to_supervised.
	
	For a univariate timeseries model, the inputs are the lagged
	target values and the output is the target value at time t presuming
	no target columns were generated in prior data processing for time t+k.
	If columns were generated for time t+k, the function will only use the last 
	column sent as the output and will put the other target columns in the 
	input (trainX) set along with the lagged inputs.  Additional processing
    of the results of series_to_supervised would be required prior to sending
    to walk_fwd_validation_rf.
	
	Parameters
	-----------
	train : training data
	testX : test data for the inputs or features

	With thanks to Jason Brownlee's Machine Learning Mastery site.
	"""

	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

def walk_forward_validation_xgb(data, n_test):
	"""Walk-forward validation for Random Forest regressor
	
	Reads in data and calls train_test_split_rows_reserved to generate
	a training set and a test set.
	
	The training set is stored as "history" for fitting the regressor.  
	It is sent to random_forest_forecast for this purpose.  One row of the
	test set is also sent to random_forest_forecast to generate a one-step
	prediction and calculate error.  This test row is then appended to
	"history" for the next iteration in fitting the regressor; it is now
	"seen data".
	
	Predictions and actual values are displayed on each iteration.  Error
	is displayed at the end of all iterations using the accumulated 
	predictions and "seen" test rows.
	
	This effectively performs a real-time refitting of the regressor as 
	new observations (test rows) come in. 
	
	Parameters
	-----------
	data : dataset prior to splitting into train and test sets
	n_test : number of rows desired for test set, taken from the end
             of the dataset

	With thanks to Jason Brownlee's Machine Learning Mastery site.
	"""
	
	predictions = list()
	train, test = train_test_split_rows_reserved(data, n_test)
	history = [x for x in train]
	for i in range(len(test)):
		testX, testy = test[i, :-1], test[i, -1]
		yhat = xgboost_forecast(history, testX)
		predictions.append(yhat)
		history.append(test[i])
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions
