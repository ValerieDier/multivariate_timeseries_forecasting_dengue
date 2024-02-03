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

## EDA

## Models

##### Acknowledgements  

I extend my appreciation to the many advisors at Lighthouse Labs for lending their expertise and providing frequent feedback.
