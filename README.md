##Ebola Prognosis Prediction Pipeline

This collection of scripts allows to train and evaluate various Machine Learning predictors 
on a provided dataset in CSV format where one of the variables is a binary response or 
output  variable we wish to predict using a subset of the remaining variables.

These scripts are meant to be used in a specific order, effectively defining a 
"prediction pipeline" that takes a number of inputs (data, variables, ranges) and outputs 
a trained predictor that can be evaluated with several metrics for model performance. This 
pipeline has been designed to facilitate systematic and reproducible model building.

The pipeline is customized to run on a dataset comprising of clinical and laboratory 
records of Ebola Virus Disease (EVD) patients, however it is completely general and can be 
applied to other dataset with virtually no modifications.

Detailed documentation and guides are available in the [wiki](https://github.com/broadinstitute/ebola-predictor/wiki).