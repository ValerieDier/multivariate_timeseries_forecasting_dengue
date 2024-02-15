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

The dataset used is supplied by Driven Data as part of a practice competition:  
DrivenData. (2016). DengAI: Predicting Disease Spread. Retrieved [February 01 2024] from https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/.

There is a training set of features (inputs), a training set of target labels (output, or the variable to predict), and a test set of features.  

The features include many environmental variables largely encompassing air temperatures, precipitation, humididty metrics, and vegetation density.  The target variable is a count
of dengue cases.  All data is recorded weekly, so one row represents a snapshot, and the next row is a snapshot taken a week later.  The weekly sampling frequency intuitively seems 
correct given the incubation time of the virus and the vectors for its spread.

![Alt Text](./img/dengue_infection_cycle_nature.jpeg)
(Dengue infection cycle image courtesy of Nature: https://www.nature.com/articles/nrdp201655)

Unfortunately, as this is set up for a competition, a test set of labels is not provided, so the test set of features cannot be used to obtain model evaluation metrics on unseen 
data.  This is addressed by splitting the training sets into training and test sets as though these were all the data provided.

## Main Steps  

1. Explore the data for quality and integrity issues
2. Examine the variables' distributions, autocorrelation on the target (time t to t-k), correlations between target and lagged features  
3. Develop an idea of the ideal time lags on which to model the timeseries  
4. Model the timeseries using a variety of model algorithms
>* Autoregression, from Statsmodels
>* Random Forest, from Scikit Learn
>* XGBoost, from XGBoost/Scikit Learn
>* LSTM (long short term memory), from Keras-Tensorflow
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
this effort.  Another measure of stationarity include the KPSS(Kwiatkowski–Phillips–Schmidt–Shin) test.

Additionally, a look at autocorrelation function (ACF) plots for the target variable can give an indication of trending or seasonality, both of which are undesirable if stationarity is a criterion
for modelling.  

The target and features do not exhibit non-stationary characteristics per the Augmented Dickey-Fuller test, and will therefore not be differenced or otherwise transformed on a
first-pass modelling basis.  As dengue fever case counts would logically change with weather patterns, and there are timeseries plots by year that support this, seasonality is 
suspected.  The assumption may be revisited, time permitting.

## Preparing Timeseries Data for Supervised Learning Algorithms  

Timeseries data must be "phrased" for supervised learning algorithms.  It amounts to lagging the data however many time intervals behind, this is known as the lag order, to 
provide the lagged feature variables (for multivariate analysis) and lagged target variable as inputs to the modelling algorithm.  This is demonstrated below.

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
>* Autoregression, from Statsmodels
>* Random Forest, from Scikit Learn
>* XGBoost, from XGBoost/Scikit Learn
>* LSTM (long short term memory), from Keras-Tensorflow

Hyperparameter tuning was trialled using GridSearchCV on the RandomForest Regressor and the XGBoost Regressor.  
SelectFromModel was used on the XGBoost Regressor.

Different lag values were attempted, somewhat supported by findings in the EDA and AutoReg trial (below), somewhat dictated by computational demands, and somewhat informed by model 
performance results.  Different train-test splits were trialled as well.  Tuning efforts were largely limited to GridSearchCV on the RandomForest and XGBoost models.  The intent was 
to trial an array of architectures to later home in on the ones offering good performance and usability.  Those models selected will then (in a future effort) be optimized by reviewing 
their hyperparameter tuning and architecture (as appropriate).

### Note on Model Fitting:  Static or Dynamic?  

In some notebooks, specifically those trialling RandomForest and XGBoost, some models were tested for their performance by sending test data row-by-row for predictions and their collection
for later execution of error calculations.  As each new test value passed through a prediction cycle, it is added to a history variable, which upon the next cycle, is used to re-fit
the model.  As each new test value becomes a "seen observation", it enriches the model by being included in the data on which the model is dynamically fit or re-trained.  

This scheme might be feasible in a setting where real-time consistent data collection occurs at a sampling frequency just high enough to allow for the re-training of the model.  This
differs from the "static" where a model is developed, deployed, monitored, and re-trained as deemed necessary.  The dynamic approach presents hazards, naturally, such as wild predictions
on erroneous data, or an inability to correctly capture a drift in the process being modelled depending on the drift's speed, behaviour, and the variables that cause the phenomenon.  
Safeguards would have to be instituted, depending on the application of the model, to ensure performance does not degrade and pose an unacceptable risk.

Where possible, model performance results are generated for both a "dynamic retraining" approach and a "static predictions" approach.

### AutoReg by Statsmodels  

This package from Statsmodels takes timeseries data "as-is": without the need to manually lag it by a number of lags "n", and produces a model fit to the training data, along 
with any parameters or error metrics desired.  This model is called an AR (autoregressive) model, and this project used it for univariate forecasting and predictions.

Autoregression requires that the data be stationary.  The EDA performed the Augmented Dickey-Fuller (ADF) test and found the data to be stationary, so no further processing was
carried out.  A set of runs could be carried out with differenced or otherwise transformed data if only to see the impact on performance.

The fitted model is then used to generate a forecast beyond the training dataset.  The predictions are collected for every step and used to generate model performance metrics.

The mean absolute error (MAE) on these models's forecasts were in the mid-twenties; this varies by the lag order used.  If new data is "fed in", using the model's coefficients
obtained during fitting, to produce one-step predictions, the model performance improves to MAEs around 6.

The classical statistical approach could be further investigated with 
the use of other models by Statsmodels such as ARMA (autoregressive moving average), or ARIMA (autoregressive integrated moving average) or ARIMAX (X = exogenous inputs) if it is 
found that differencing for stationarity is required.  

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

### LSTM (long short-term memory) by Keras-Tensorflow  

Long short-term memory networks are a special type of RNN (recurrent neural network).  The diagram below, something that resembles stereo assembly instructions from the late twentieth
century, may help visualize how it works:  

![Alt Text](./img/LSTM3-chain.png)
(LSTM diagram image, and conceptual understanding, courtesy of colah's blog: https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  

Briefly put, the LSTM has repeating modules with information passing through - the cell "state" - and the inner workings depicted in the diagram represent various "gates" that act
a bit like valves (a nod to my chemical engineering background ;) but using functions like sigmoids or tanh (hyperbolic tangent function) to decide how much information to pass
through.  This impacts the updates to the cell state as it flows through a module, and thus impacts what might be imagined as "memory".

Now, how to configure:  

"Configuring neural networks is difficult because there is no good theory on how to do it."  
(wisely offered by https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)  

Observations to date include the following properties that can be adjusted:  
* Number of layers  
* Number of neurons (times number of layers - have fun)  
* Number of epochs (1 epoch = 1 full run-through of the training data), a hyperparameter  
* Batch size (nothing to do with cookies :cookie: )   
* Activation functions (unclear if modifiable for LSTM)  

Wondering what the difference between a batch and an epoch is?  An epoch, as noted above, is one full cycle through the entire training set.  An epoch is made of one or more batches.
Batches are used to indicate how much data the training algorithm can use to update model parameters.  A batch can take on various sizes as defined by the number of samples (rows 
of data) taken from the training set.  It can be comprised of the entire training set, in which case the model parameters are said to be trained by batch gradient descent.  If the
batch is made up of one row from the training set, the model parameters are said to be trained by stochastic gradient descent.  If the batch size is greater than one row but less 
than the entire training set, the training process is called mini-batch gradient descent.  

For a more thorough discussion:  https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/  

The same source that offered wisdom above provides a tutorial for evaluating different network properties, which will be re-consulted for further development of the LSTM model attempted 
in the lstm_trial_nlags_MV.ipynb (and its 1lag sibling) notebook.  
https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/  

Adding to the treasure chest is the option to configure LSTMs to be stateful or stateless, with impacts to information remembered through an LSTM network.  This is a quest for another
day.  

## Model performance  

Mean absolute error is an error metric often used for its interpretability: it's in the original units of the predicted variable.  This can make for some cumbersome inversions if scaling
was used to preprocess the inputs to a model.  It is a typical error used in modelling competitions, though MAPE (mean absolute percentage error) occasionally makes an appearance for
its interpretability as a precent, rather than units of measure.  

## Challenges  

* The dataset is likely on the short end of acceptable.  More rows of data, obtained simply from collecting in the same area for more years, offers more opportunity to train models 
and try different configurations.  
* There were variables that seemed almost the same, including many temperature variables.  Much more investigation is required to understand their inclusion.  
* Producing correctly formatted timeseries data for supervised learning takes many steps and checks.  It's time-consuming even when using another's template, especially if it's 
being adapted to a different situation.  
* Burden of choice: in a time when modelling approaches can be trialled very quickly across a broad swath of algorithms, a better methodology for choosing a modelling framework is
sorely needed.  This project was an early prospecting endeavour, and by no means an exhaustive review of all options.  There is much to explore from here.  

## Development Work  
### Partial Autocorrelation  
Another tool in the timeseries modelling toolbox is partial autocorrelation.  This takes into account the relationships between all variables, rather than just considering the
relationship between two variables of interest.  Further study is required to apply.  

Statistics by Jim often delivers a good starting point:  https://statisticsbyjim.com/time-series/autocorrelation-partial-autocorrelation/  

### Feature Selection  
The focus of this project was to demonstrate how timeseries data can be prepared for modelling, as well as gather model performance results from using different frameworks. There 
was therefore less emphasis on selecting key features for the model in the EDA, and more effort spent on examining relationships between the target and the lagged variables
through correlation and autocorrelation calculations.  The features could be revisited to see which ones to keep, though as the data must be phrased into a supervised learning 
problem, the analysis for feature selection is further complicated by the consideration of lagged variables.  That is, some features may be more impactful *if* lagged by some 
interval k. Exploration of timeseries forecasting practices should bring insights.  The use of SelectFromModel in the XGBoost notebook provided a convenient feature selection tool
that warrants further exploration.  Its results suggested a considerable paring down of the inputs might still deliver good model performance.  

### Model Algorithms: Which to Choose?  

There are many papers on every type of model one can have (or not yet) heard of, for use in a multitude of use cases.  Timeseries forecasting has its own niche, but one can 
imagine the different types of challenges encountered in timeseries forecasting for different situations.  

General Review of Models Used for Timeseries Modelling:  
https://www.mdpi.com/2078-2489/14/11/598  

The paper is a review of a large number of modelling algorithms.  Factoring in the various approaches that can be used to prepare the data for timeseries modelling, the number of 
options quickly becomes large.  

From this paper:  "The short-term memory of recurrent networks is one of their major drawbacks and one of the main reasons why attention mechanisms and Transformers were originally 
introduced in deep learning (see Section 4.1)."  

The approach has already evolved past a "simple" LSTM, making it an introduction to deep learning for timeseries, but advances have been made in this area to newer architectures that
warrant investigation.  

### "Ready-to-Serve" Packages  

An exploration of packages already configured for timeseries, such as the suite of Statsmodels statistical modelling options, but in machine learning frameworks, would be instructive.
Tensorflow does allow for the use of timeseries data that has not been "phrased for" supervised learning, that is, that has not been lagged and concatenated into a wide dataframe. 
However, algorithmslike RandomForest, which makes use of bootstrapping, could not be used on data that has not been lagged for row-by-row input to a model.  The random sampling would 
lose the temporal information that is only maintained by respecting the sequence of rows and their values in time.  

There are packages like pyts that offer timeseries classification.  

There are also other modelling packages such as Facebook's Prophet, and likely many other open-source timeseries modelling packages offered.  

For an example of Tensorflow's treatment of timeseries:  https://www.tensorflow.org/tutorials/structured_data/time_series  

An interesting "first quick pass" at comparing machine learning algorithms for timeseries could be PyCaret's regression module.  It effectively runs many different algorithms on the
data supplied, with little upfront configuration, and delivers various model error metrics.  One could choose a small subset of these algorithms to investigate and tune further from 
there.  An example is supplied here:  https://www.datacamp.com/tutorial/tutorial-time-series-forecasting.  The tutorial discussed many other timeseries modelling packages available.
Already the variety is growing.  

### State Space Modelling  

State space modelling is a system identification technique buried under a couple decades of disuse (mine).  It aims to describe a system with unobserved variables.  

Some general theory:  https://www.mathworks.com/help/ident/ug/what-are-state-space-models.html  

One package offering the needed architecture:  https://www.statsmodels.org/dev/statespace.html  

This will be revisited as early introductions to the topic were driven from mechanistic models and their differential equations to describe system dynamics.  This describes their 
behaviour through time.  

Their use in timeseries modelling comes up occasionally and one could benefit from familiarization.  

### Refactoring and Automation  
There are still many pieces of code that are repeated throughout the notebooks, and a number of areas that are error-prone, such as the production of models later used to optimize 
hyperparameter tuning.  Putting these in functions residing in modules would allow for standardization and reduce the risk of confusing models.  

### Pipelining  

Using the pipeline functions readily available from Scikit Learn could greatly simplify the production of models provided the data preprocessing doesn't pose problems.  In the case of
LSTM it could remove some complications around the inversion of scaling to return results back to original units, though it looks like building a Keras/Tensorflow model to use in
pipelines brings its share of extra configuration:  https://queirozf.com/entries/scikit-learn-pipeline-examples#keras-model  

## Acknowledgements  

I extend my appreciation to the many advisors at Lighthouse Labs for lending their expertise and insights.  

The extensive templates found on https://machinelearningmastery.com/ were invaluable to the development of these notebooks.  Approaches from concatenated lagged data to construct
a dataframe suitable for timeseries modelling using supervised learning, to sending test data row-by-row for dynamic model refitting, and likely many more strategies were gathered
from the author's website, with a few adaptations made as necessary.  

ChatGPT enabled much faster automation of plotting, as well as merging of variable names to nameless features when model outputs were shorn of their human-friendly details.
Troubleshooting was also aided by prompting and careful review of suggestions.  I learned from it, and it learned from me.  Its help with syntax and functions I'd never heard of
allows for energy and time to be redirected to analysis, critical review, and a broader systems-view on modelling methodology and operationalization.  

Professors from my university years provided the canvas for my interest in system identification.  

The advent of large-scale, but more importantly, *regular and frequent* collection of data supplies the possibility of predicting any number of events that, given adequate notice,
can be handled more smoothly than with little or no preparation.  The real challenge is in balancing these possibilities against the risks of poor forecasting and the potential for
irresponsible collection, storage, and use of data.  

