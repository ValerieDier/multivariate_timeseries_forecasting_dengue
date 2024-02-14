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

Unfortunately, as this is set up for a competition, a test set of labels is not provided, so the test set of features cannot be used to obtain model evaluation metrics on unseen 
data.  This is addressed by splitting the training sets into training and test sets as though these were all the data provided.

## Main Steps  

1. Explore the data for quality and integrity issues
2. Examine the variables' distributions, autocorrelation on the target (time t to t-k), correlations between target and lagged features  
3. Develop an idea of the ideal time lags on which to model the timeseries  
4. Model the timeseries using a variety of model algorithms
* Autoregression, from Statsmodels
* Random Forest, from Scikit Learn
* XGBoost, from XGBoost/Scikit Learn
* LSTM (long short term memory), from Keras-Tensorflow
5. Evaluate model performance metrics between models, revise lags and hyperparameters as necessary
6. Select the best model developed
7. Re-think the features: can we apply a select_features function compatible with the model algorithm with the best results so far? Can it point to better feature selection?
8. Consider changing the features, re-running the model
7. Optimize the best model so far either through hyperparameter tuning (consider tools like Grid or RandomSearchCV)
8. Run the entire dataset through the best model to get the best possible fit and performance metrics

Note: traditional feature selection as is normally carried out in EDA is given less importance in this project only to allow for the development of many models and to account for
the time required to preprocess the timeseries data.  Addtionally, a "live modelling re-fit" scheme, templated from the Machine Learning Mastery site, is elaborated in some of the
model algorithms tested, which proved to require much time.  Feature selection is re-approached in the XGBoost notebook using a Scikit Learn function.  There is also Lasso regression
that can provide some support for feature selection.  This was not attempted here in favour of trying a broad array of learning algorithms.

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

This is done for the target variable to regress against its lagged self, and for the features (the environmental variables) to regress against their lagged values.

## Models

The model frameworks trialled include:
* Autoregression, from Statsmodels
* Random Forest, from Scikit Learn
* XGBoost, from XGBoost/Scikit Learn
* LSTM (long short term memory), from Keras-Tensorflow

Hyperparameter tuning was trialled using GridSearchCV on the RandomForest Regressor and the XGBoost Regressor.  
SelectFromModel was used on the XGBoost Regressor.

### Note on Model Fitting:  Static or Dynamic?  

In some notebooks, specifically those trialling (X, Y, Z - check), some models were tested for their performance by sending test data row-by-row for predictions and their collection
for later execution of error calculations.  As each new test value passed through a prediction cycle, it is added to a history variable, which upon the next cycle, is used to re-fit
the model.  As each new test value becomes a "seen observation", it enriches the model by being included in the data on which the model is dynamically fit or re-trained.  

This scheme might be feasible in a setting where real-time consistent data collection occurs at a sampling frequency just high enough to allow for the re-training of the model.  This
differs from the "static" where a model is developed, deployed, monitored, and re-trained as deemed necessary.  The dynamic approach presents hazards, naturally, such as wild predictions
on erroneous data, or an inability to correctly capture a drift in the process being modelled depending on the drift's speed and behaviour.  Many guardrails and checks would have to be
instituted, depending on the application of the model, to ensure performance does not degrade heavily *and* quickly due to this dynamic updating scheme.

Where possible, model performance results are shown for both a "dynamic retraining" approach and a "static predictions" approach.

### AutoReg by Statsmodels  

This package from Statsmodels takes timeseries data "as-is", that is, without the need to manually lag it by a number of lags "n", and produces a model fit to the training data, along 
with any parameters or error metrics desired.  This model is called an AR (autoregressive) model.

Autoregression requires that the data be stationary.  The EDA performed the Augmented Dickey-Fuller (ADF) test and found the data to be stationary, so no further processing was
carried out.

The fitted model is then used to generate a forecast beyond the training dataset.  The predictions are collected for every step and used to generate model performance metrics.

The mean absolute error (MAE) on these models were in the mid-twenties; this varies by the lag order used.  The classical statistical approach could be further investigated with 
the use of other models by Statsmodels such as ARMA (autoregressive moving average), or ARIMA (autoregressive integrated moving average) if it is found that differencing for stationarity
is required.


### Random Forest by Scikit Learn  

Supervised learning algorithms such as Random Forest are built to accommodate time-invariant data:  you have inputs collected on a patient, customer, or item; you have outputs regarding
that patient's health or that customer's purchases, and you want predictions on the patient's diagnostic status or that customer's likelihood of returning to a given retailer.  This 
means timeseries data must be "phrased" as a supervised learning problem by lagging the data, targets and features in the multivariant case, by however many lags you wish to experiment
with.  The data is generated, collated, and undesired columns of data (i.e. unlagged features) must be carefully removed for proper input to the algorithm.  

Random Forests use smaller decision trees built with bootstrapped data (sampling with replacement) and random subsets of features to split on to come up with an estimator that strikes 
the right balance between low model bias and low variance.  Predictions on regressors are averaged, while labels on classifiers are chosen on highest vote.

The RandomForest models trialled used 1000 estimators.  The GridSearchCV executed on the multivariate models optimizes over hyperparameters such as the number of estimators, and the
maximum depth of trees built, and so on.  The grid that is specified for GridSearchCV can be customized to a given project.

The univariate RandomForest model trials had MAEs largely in the 6-9 range, while the multivariate models had MAEs still in the 6-9 range.


### XGBoost by XGBoost/Scikit Learn  

XGBoost is an ensemble method like RandomForest, but it uses the residuals, or errors, from previously-built trees as inputs to the next trees to improve upon the performance of the
estimator.  It runs very quickly, and predicts to impressive accuracy; keeping a vigilant eye to overfitting may be the biggest concern.  GridSearchCV was also used on the XGBoost
regressor to tune hyperparameters.  A brief exploration of SelectFromModel was performed with an XGBoost model.

MAE values were nearer to 7-9 for the XGBoost Regressors without dynamic updating; these fell to 3.5 and less for dynamically trained models.

### LSTM (long short term memory) by Keras-Tensorflow  

(more: structure/hyperparams, train/test, etc - see notebook. Change test set size and collect MAEs again.)

## Model performance  

Mean absolute error is an error metric often used for its interpretability: it's in the original units of the predicted variable.  This can make for some cumbersome inversions if scaling
was used to preprocess the inputs to a model.  (more?)

## Challenges  

* There were variables that seemed almost the same, including many temperature variables.  Much more investigation is required to understand their inclusion.
* Producing correctly formatted timeseries data for supervised learning takes many steps and checks.  It's time-consuming even when using another's template, especially if it's 
being adapted to a different situation.
*


## Development Work  
### Partial Autocorrelation  
Another tool in the timeseries modelling toolbox is partial autocorrelation.  This takes into account the relationships between all variables, rather than just considering the
relationship between two variables of interest.  Further study is required to apply.

### Feature Selection  
The focus of this project was to demonstrate how timeseries data must be prepared for modelling, as well as gather model performance results from using different frameworks.  
There was therefore less emphasis on selecting key features for the model in the EDA, and more effort spent on examining relationships between the target and the lagged variables
through correlation and autocorrelation calculations.  The features could be revisited to see which ones to keep, though as the data must be phrased into a supervised learning 
problem, the analysis for feature selection is further complicated by the consideration of lagged variables.  That is, some features may be more impactful *if* lagged by some 
interval k. Exploration of timeseries forecasting practices should bring insights.  The use of SelectFromModel in the XGBoost notebook provided a convenient feature selection tool
that warrants further exploration.  Its results suggested a considerable paring down of the inputs might still deliver good model performance.  

### State Space Modelling  

(comments and links)

### Model Algorithms: Which to Choose?  

There are many papers on every type of model one can have heard (or not yet) heard of, for use is a multitude of sectors, industries, and services.  Timeseries forecasting has its
own niche, but one can imagine the different types of challenges encountered in timeseries forecasting for different situations.

General Review of Models Used for Timeseries Modelling:
https://www.mdpi.com/2078-2489/14/11/598

The paper is a review of a large number of modelling algorithms.  Factoring in the various approaches that can be used to prepare the data for timeseries modelling, the number of options
quickly becomes large.  

From this paper:  "The short-term memory of recurrent networks is one of their major drawbacks and one of the main reasons why attention mechanisms and Transformers were originally 
introduced in deep learning (see Section 4.1)."

The approach has already evolved past a "simple" LSTM, making it an introduction to deep learning for timeseries, but not a canditate for a model.

### Ready-to-Serve Packages  

An exploration of packages already configured for timeseries, such as the suite of Statsmodels statistical modelling options, but in machine learning frameworks, would be instructive.
Tensorflow does allow for the use of timeseries data that has not been "phrased for" supervised learning, that is, lagged and concatenated into a wide dataframe.  However, algorithms
like RandomForest, which makes use of bootstrapping, could not be used on data that has not been lagged for row-by-row input to a model.  The random sampling would lose the temporal
information that is only maintained by respecting the sequence of rows and their values in time.

There are packages like pyts that offer timeseries classification.  Similar tools must exist or are being developed for timeseries regression.

For an example of Tensorflow's treatment of timeseries:  https://www.tensorflow.org/tutorials/structured_data/time_series  

### Refactoring and Automation
There are still many pieces of code that are repeated throughout the notebooks.  Putting some of these in functions residing in modules would allow for standardization.


### Pipelining (?)

(comment)


## Acknowledgements  

I extend my appreciation to the many advisors at Lighthouse Labs for lending their expertise and insights.  

The extensive templates found on https://machinelearningmastery.com/ were invaluable to the development of these notebooks.  Approaches from concatenated lagged data to construct
a dataframe suitable for timeseries modelling using supervised learning, to sending test data row-by-row for dynamic model refitting, and likely many more strategies were gathered
from the author's website, with a few adaptations made as necessary.  

ChatGPT enabled much faster automation of plotting, as well as merging of variable names to nameless features when model outputs were shorn of their human-friendly details.
Troubleshooting was also aided by prompting and careful review of suggestions.  

Professors from my university years provided the canvas for my interest in system identification, and thinking about modelling in general.  

The advent of large-scale but more importantly, *regular and frequent* collection of data supplies the possibility of predicting any number of events that, given adequate notice,
can be handled more smoothly than with little or no preparation.  The real challenge is in balancing these possibilities against the risks of poor forecasting and the potential for
irresponsible collection, storage and use of data.

