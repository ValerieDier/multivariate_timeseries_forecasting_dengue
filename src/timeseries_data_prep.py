from pandas import DataFrame, concat

def series_to_supervised_uv(data, n_in=1, n_out=1, dropnan=True):
	"""For univariate timeseries data.  
	
	Reads in a dataframe and the n_in parameter as a lag or order value
	for autoregresion.
	
	Reads in the data and shifts it by one row down to obtain a variable
	value at time t-1.  A column is appended to the result.
	
	The number of times this is done corresponds to n_in.
	
	Then the original variable is appended without shifting.  If a value
	was specified for n_out (default 1), the variable will then be shifted
	up a row and appended to the result.  This is done as many times as is
	specified by n_out.
	
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