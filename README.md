# Multivariate Timeseries Forecasting  

## Motivation  

This notebook leverages multivariate timeseries modelling frameworks to forecast the spread of dengue.  The motivation for this exercise is to explore the statistical and machine
learning algorithms available for the prediction of a target variable over time, particularly when features or independant variables are provided to help establish causal relationships. 

> This work is applicable to a broad array of sectors:
> * Maintenance and reliability of equipment  
> * Manufacturing process upsets or incidents  
> * Occupational health incidents  
> * Public health crises  
> * Security incidents  
> * Supply chain or logistic disruptions 
> * Financial default 
> * (...)

## Data Overview  

The dataset used is supplied by Driven Data as part of a practice competition (include citation here).  There is a training set of features (inputs), a training set of labels
(output, or the variable to predict), and a test set of features.  

The features include many environmental variables largely encompassing air temperatures, precipitation, humididty metrics, and vegetation density.  The target variable is a count
of dengue cases.

Unfortunately, as this is set up for a competition, a test set of labels is not provided, so the test set of
features cannot be used to obtain model evaluation metrics on unseen data.  This is addressed by splitting the training sets into training and test sets as though these were all
the data provided.

## Main Steps  

1. Explore the data for quality and integrity  
2. Examine the variables' distributions, autocorrelation on the target (time t to t-k), correlations between target and lagged features  
3. Develop an idea of the ideal time lags on which to model the timeseries  
4. Model the timeseries using autoregression, (...tbd)  
5. Evaluate model performance metrics between models   
6. Select the best model developed   
7. Run the test features through the best model to obtain predictions on dengue case counts  

## EDA  

Histograms and timeseries plot provide an idea of stationarity of the data, that is, whether the data's statistics (e.g. mean, variance) change with time.  Classical statistical
forecasting methods require stationary data.  When data is found to be non-stationary, it can be differenced or log transformed.  Differencing is simply subtracting the data at
time t-1 from the data at time t.

The Augmented Dickey-Fuller test can be applied to the data to establish whether the data is stationary.  It will produce the statistic, critical values, and a p-value to support
this effort.  

Additionally, a look at autocorrelation plots for the target variable can give an indication of trending or seasonality, both of which are undesirable if stationarity is a criterion
for modelling.  

The target and features do not exhibit non-stationary characteristics per the Augmented Dickey-Fuller test, and will therefore not be differenced or otherwise transformed on a
first-pass modelling basis.  As dengue fever case counts would logically change with weather patterns, which would represent seasonality, this may be revisited, time permitting.

## Models

(working on it)

##### Acknowledgements  

I extend my appreciation to the many advisors at Lighthouse Labs for lending their expertise and insights.
