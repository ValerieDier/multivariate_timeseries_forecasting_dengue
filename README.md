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

What might normally be presented as a 'time invariant' data set, such as health markers with the intent to predict if a person has heart disease, could be configured as a timeseries
when data is collected on a regular basis over time.

Imagine a given patient's health markers being collected regularly over the course of a decade prior to the age where heart disease is normally detected.  One could then attempt
to predict, ahead of time, by looking back in time, if the patient is at higher risk than their peers for developing heart disease.  Preventative actions, proportional to the risk, 
could be taken to reduce the risk of heart disease developing.  

Similar perpectives may be taken in business, industry, and the public sector.

## Data Overview  

The dataset used is supplied by Driven Data as part of a practice competition (include citation here).  There is a training set of features (inputs), a training set of target labels
(output, or the variable to predict), and a test set of features.  

The features include many environmental variables largely encompassing air temperatures, precipitation, humididty metrics, and vegetation density.  The target variable is a count
of dengue cases.  All data is recorded weekly, so one row represents a snapshot, and the next row is a snapshot taken a week later.

Unfortunately, as this is set up for a competition, a test set of labels is not provided, so the test set of
features cannot be used to obtain model evaluation metrics on unseen data.  This is addressed by splitting the training sets into training and test sets as though these were all
the data provided.

## Main Steps  

1. Explore the data for quality and integrity issues
2. Examine the variables' distributions, autocorrelation on the target (time t to t-k), correlations between target and lagged features  
3. Develop an idea of the ideal time lags on which to model the timeseries  
4. Model the timeseries using a variety of model algorithms
* Autoregression, from Statsmodels
* Random Forest, from Scikit Learn
* XGBoost, from XGBoost/Scikit Learn
* LSTM (long short term memory), from Keras-Tensorflow
5. Evaluate model performance metrics between models, revise as necessary
6. Select the best model developed   
7. Optimize the best model
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
first-pass modelling basis.  As dengue fever case counts would logically change with weather patterns, and there are timeseries plots by year that support this, seasonality is 
suspected.  The assumption may be revisited, time permitting.

## Preparing Timeseries Data for Supervised Learning Algorithms  

Timeseries data must be "phrased" for supervised learning algorithms.  It amounts to lagging the data however many time intervals behind, this is known as the lag order, to 
provide the lagged input variables and lagged target variable as features to the modelling algorithm.  This is demonstrated below.

The following is the first row of data from the dataset in its unlagged and then lagged form:  

Raw total_cases:  
[[ 4.00000000e+00  1.22600000e-01  1.03725000e-01  1.98483300e-01  1.77616700e-01  2.97572857e+02  2.97742857e+02  2.92414286e+02  2.99800000e+02  2.95900000e+02  3.20000000e+01  7.33657143e+01  1.24200000e+01  1.40128571e+01  2.62857143e+00]

Lagged total_cases:  
[[ **4.00000000e+00**  1.22600000e-01  1.03725000e-01  1.98483300e-01  1.77616700e-01  2.97572857e+02  2.97742857e+02  2.92414286e+02  2.99800000e+02  2.95900000e+02  3.20000000e+01  7.33657143e+01  1.24200000e+01  1.40128571e+01  2.62857143e+00  **5.00000000e+00**]

The input sequence on total_cases, unmodified, is 4 5 4 3 6 2 4 (just the first few values of the univariate series).  

In lagging this (y(t)), ignoring other features, you get:  

| y(t-1)      | y(t) |
| ----------- | ----------- |
| NaN         | 4           |
| 4           | 5           |
| 5           | 4           |
| 4           | 3           |
| 3           | 6           |
| 6           | 2           |
| 2           | 4           |
| 4           | NaN         |

The rows with NaNs are elimitated, so you end up with:

[[4 5]  
 [5 4]  
 [4 3]  
 [3 6]  
 [6 2]  
 [2 4]]  

 If everything was done correctly, you should end up with an input sequence to your modelling engine that puts your lagged columns first. Your unlagged output will be last.

 Therefore, the very first row going to modelling should have 4 first, as the lagged target (y(t-1)), and a 5 last as the unlagged target (y(t)).  That's what is seen above in
 "Lagged total_cases:", with the appropriate values in bold.  

## Models

The model frameworks trialled include:
* Autoregression, from Statsmodels
* Random Forest, from Scikit Learn
* XGBoost, from XGBoost/Scikit Learn
* LSTM (long short term memory), from Keras-Tensorflow

### AutoReg by Statsmodels  

This package from Statsmodels takes timeseries data "as-is", that is, without the need to manually lag it by a number of lags "n", and produces a model fit to the training data, along with
any parameters or error metrics desired.  

(more: structure/hyperparams, train/test, etc - see notebook)

### Random Forest by Scikit Learn  

Supervised learning algorithms such as Random Forest are built to accommodate time-invariant data:  you have inputs, you have outputs, you want predictions.  This means timeseries data
must be "phrased" as a supervised learning problem by lagging the data, targets and features in the multivariant case, by however many lags you wish to experiment with.  The data is generated,
collated, and undesired columns of data (i.e. unlagged features) must be carefully removed for proper input to the algorithm.  

Random Forests use smaller decision trees built with bootstrapped data (sampling with replacement) and random subsets of features at the splits to come up with an estimator that strikes the
right balance between model bias and variance.  Predictions on regressors are averaged, labels on classifiers are chosen on highest vote.

(more: structure/hyperparams, train/test, etc - see notebook)

### XGBoost by XGBoost/Scikit Learn  

(comments - see notebook)

### LSTM (long short term memory) by Keras-Tensorflow  

(more: structure/hyperparams, train/test, etc - see notebook)

## Model performance  



## Challenges  

* There were variables that seemed almost the same, including many temperature variables.  
* 

## Development Work  
### Partial Autocorrelation  
Another tool in the timeseries modelling toolbox is partial autocorrelation.  This takes into account the relationships between all variables, rather than just considering the
relationship between two variables of interest.  Further study is required to apply.

### Feature Selection  
The focus of this project was to demonstrate how timeseries data must be prepared for modelling, as well as gather model performance results from using different frameworks.  
There was therefore less emphasis on selecting key features for the model, and more effort spent on examining relationships between the target and the lagged variables through 
correlation and autocorrelation calculations.  The features could be revisited to see which ones to keep, though as the data must be phrased into a supervised learning problem, 
the analysis for feature selection is further complicated by the consideration of lagged variables.  That is, some features may be more impactful *if* lagged by some interval k. 
Exploration of timeseries forecasting practices should bring insights.

### Refactoring  
There is still a good number of pieces of code that are repeated throughout the notebooks.  Putting some of these in functions residing in modules would allow for standardization.

### Pipelining (?)

(comment)


## Acknowledgements  

I extend my appreciation to the many advisors at Lighthouse Labs for lending their expertise and insights.
The extensive templates found on https://machinelearningmastery.com/ were invaluable to the development of these notebooks.