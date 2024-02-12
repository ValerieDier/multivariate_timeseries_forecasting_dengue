from pandas import DataFrame, concat

def series_to_supervised_uv(data, n_in=1, n_out=1, dropnan=True):
	"""For univariate timeseries data.  
	
	Reads in a dataframe and the n_in parameter as a lag or order value
	for autoregresion.
	
	The data is first shifted by however many rows as specified by n_in.
	The column is appended to a list.  The data is then shifted by however
	many rows correspond to "n_in - 1", and the column is appended to the list.
	This continues, shifting by one less row each time, until the data is only 
	shifted by one row and the column is appended.
		
	Then the original unlagged variables are appended without shifting.  If a 
	value was specified for n_out (default 1), the variable will then be shifted
	up a row and appended to the result.  This is done as many times as is
	specified by n_out.  This will result in variables at time t+k (k taking on
	values from 1 to n_out).

	Caution:

	walk_fwd_validation_rf will call upon random_forest_forecast, which in turn
	will assign the very last column from the results returned by series_to_supervised,
	or any processing the user applies after receiving these results, as the output 
	variable to predict.  If n_out is assigned a value other than 1 in this function,
	additional processing will be required prior to sending the data to walk_fwd_validation_rf.

	Parameters
	-----------
	data : a column from a dataframe sent as a list
	n_in : number of lags desired, t-n_in (autoregression order), default is 1
	n_out : number of 'forward lags' desired, t+n_out, default is 1
	dropnan : default is True
	"""

	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

def series_to_supervised_mv(data, n_in=1, n_out=1, dropnan=True):
	"""For multivariate timeseries data.  
	
	Reads in a dataframe and the n_in parameter as a lag or order value
	for regresion.
	
	The data is first shifted by however many rows as specified by n_in.
	The column is appended to a list.  The data is then shifted by however
	many rows correspond to "n_in - 1", and the column is appended to the list.
	This continues, shifting by one less row each time, until the data is only 
	shifted by one row and the column is appended.
		
	Then the original unlagged variables are appended without shifting.  If a 
	value was specified for n_out (default 1), the variable will then be shifted
	up a row and appended to the result.  This is done as many times as is
	specified by n_out.  This will result in variables at time t+k (k taking on
	values from 1 to n_out).

	Caution:

	walk_fwd_validation_rf will call upon random_forest_forecast, which in turn
	will assign the very last column from the results returned by series_to_supervised,
	or any processing the user applies after receiving these results, as the output 
	variable to predict.  If n_out is assigned a value other than 1 in this function,
	additional processing will be required prior to sending the data to walk_fwd_validation_rf.

	There may be an opportunity to verify if this can be used for the univariate case
	without undue editing to other functions or calls to accomodate any change in the
	resulting data returned.
	
	Parameters
	-----------
	data : a column from a dataframe sent as a list
	n_in : number of lags desired, t-n_in (autoregression order), default is 1
	n_out : number of 'forward lags' desired, t+n_out, default is 1
	dropnan : default is True
	"""

	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def train_test_split_rows_reserved(data, n_test):
	"""Performs a split of the data that keeps the last n_test rows of the dataset
	for a test set.  Returns a training set "M - n_test" rows by N columns, and a 
	test set n_test rows by N columns.

	Parameters
	-----------
	data : a dataframe or list of arrays M rows by N columns
	n_test : number of rows to keep for the test set
	"""

	return data[:-n_test, :], data[-n_test:, :]