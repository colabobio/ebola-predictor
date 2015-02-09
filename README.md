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

###Basic concepts

The pipeline roughly involves three stages: defining training and test sets, fitting the
desired predictor using the training set, and evaluating the predictor on the test set. The 
pipeline includes a number of built-in predictors (a Decision Tree and a Neural Network), 
but  additional predictors can be added to it by following a few coding conventions. 

Because missing values is a typical problem in survey and health data, the pipeline allows 
to "complete" an incomplete training set by using a variety of methods. Some of them are 
already built into the pipeline, but additional methods can be implemented also by following
a number of conventions.

Let's go over a simple usage case in order to exemplify how all these stages work together.

**1) Model preparation.** This preliminary stage involves setting up the source data file, 
the variables to include in the predictor, restricting the data to a specific subset of 
interest, and a few other adjustments. 

The source dataset must be contained in a standard CSV file where each column holds a separate 
variable and each row represents a distinct data sample. The first row in the data file must 
contain the names of the variables stored in each column. The location of the data file must 
be indicated in the file _sources.txt_ inside the _data_ folder. This location can be relative 
or absolute, for example the default sources file simply points to a csv file already inside the
_data_ folder:

```
./data/data.csv
```

The list of variables to include in the predictive model are listed in the _variables.txt_
file:

```
OUT category
PCR float
TEMP float
DIARR category
AST_1 float
Cr_1 float
```

Each row in this file must contain two values separated by a space: the variable name (as 
it appears in first row of the CSV file), and the type. The supported types are category 
(for nominal or ordinal variables), and int and float (for numerical variables). The first 
row in the variables file *must* be response variable to predict, which must also be not 
only of category type, but also binary, and in particular adopting the values 1 and 0. 
This limitation currently applies to any other category variable to include in the model.

In order to restrict the application of the model to a subset of the entire data, ranges
can be defined on the variables. This ranges are set in the *ranges.txt* file:

```
DIAG category 1
AGE int 0,50
```

In this case, only rows where the DIAG category variable adopts the value 1 and AGE is 
between 0 and 50 will be used to construct the training and test sets.

As mentioned before, missing values can be handled by the pipeline using different methods
(see next section). However, it is required that the missing values in the source CSV file 
are identified by the "\N" string (backslash + N character).

Finally, some evaluation methods display labels for the two possible values of the outcome 
variable. These labels can be set in the labels.txt file as follows:

```
0 Discharged
1 Died
```

**2) Creating training/test sets.** Once the variables and other model settings are defined,
a pair of training/test sets can be constructed. This can be achieved using the *utils/makesets.py*
script:

```bash
python utils/makesets.py
```

The output of the execution of this command should be two files created inside the *data* 
folder, *training-data.csv* and *testing-data.csv*. The script will assign a percentage of 
the **complete** rows (without missing values) to the test file. The rest of the rows, 
including those with missing entries, will be stored in the training file. By default, the
percentage is 50%, but this value, as well as the name of the files, can be set using the 
following command line arguments:

```bash
makesets.py [-h] [-t TRAIN] [-T TEST] [-p PERCENTAGE]
```

The -h flag can be used to get help about each argument, and in general all scripts in 
the pipeline support the help flag. So, to store 70% of the complete rows in the test set, 
and the remaining 30% plus any other rows containing missing values one would use:

```bash
makesets.py -p 70
```

Since a prediction algorithm will, in general, require a training set without missing values,
an additional step is required to either remove missing values or impute them using some 
algorithm. The pipeline provides two built-in methods to handle missing values: list-wise 
deletion, and Multiple Imputation using the [Amelia](http://gking.harvard.edu/amelia) 
package for R.

List-wise deletion (where a row is removed if it has at least one missing entry) can be 
applied using the *utils/listdel.py* script, which by default will read the original 
training set file from write the training set 
with the missing values removed to *data/training-data-completed.csv*:

```bash
python utils/listdel.py [-h] [-i INPUT] [-o OUTPUT]
```

Amelia, in contrast, generates m values for each missing entry and thus it can create m 
completed datasets. These imputed values are drawn from a multivariate Gaussian distribution 
that it is estimated by an Expectation-Maximization algorithm from the known data. The 
*data/amelia.py* script calls the Amelia routines in R through the [RPy2 package](http://rpy.sourceforge.net/) 
(read the dependencies for more details on the dependencies required by the pipeline), and
by default it will generate 5 imputed datasets, wich are then aggregated to create an 
augmented training set without missing values. The known values in the original file are 
not affected and are the same across all the imputed datasets. The default filenames of the 
amelia scrip are the same as with listdel. However, it accepts several additional arguments
that can be used to control the imputation procedure:

```bash
python utils/amelia.py [-h] [-i INPUT] [-o OUTPUT] [-n NUM_IMPUTED] [-r NUM_RESAMPLES] [-c] [-p]
```

* -n, --num_imputed: sets the number of imputed datasets to aggregated into the completed training set file
* -r, --num_resamples: sets the number of resample until a value inside the bounds defined for each variable is found
* -c, --in_check: enables checking the input data to detect highly co-linear variables
* -p, --get_plots: enables the genration of plots characterizing the imputation results, which are saved to the *out* folder.
  This plots include a missingness map, comparisons between the observed and imputed densities, and the quality of imputation
  plot (only for numerical variables)

For example, to generate a training set aggregating 10 imputed datasets and save the plots, 
one would do:

```bash
python utils/amelia.py -n 10 -p
```

In order to get a quick sense of the scale of missingness for the selected variables and ranges, 
the *missing.py* script provides the counts and percentage of missing values for each variable:

```bash
python utils/missing.py
```

However, the missingness map generated by Amelia is probably more useful since it gives a 
visual idea of the amount of missingness across variables.





###Dependencies
* Pandas
* Numpy
* matplotlib
* scikit-learn

* RPy2
* R (Amelia II, PredictABEL)
* pydot, Graphviz



###Plotting
utils/scatterplot.py





##Reference prediction: EPS

Use either variables-eps7 or variables-eps10, set ranges to all ages then:

```bash
python utils/makesets.py -p 100
```

```bash
python eps/eval.py --cutoff 0 --method report
```
cutoff sets the EPS score above which a patient is predicted to die.





##Batch mode

1) Create training/test sets

```bash
python init.py -n [number of iterations] -t [test percentage] -i [number of imputed files to run MI on] -s [starting id]
```

```bash
python init.py -n 100 -s 0 -t 50 -m amelia [num_imputed=5 num_resamples=10000 --in_check --gen_plots]
```

2) Train the predictors

```bash
python train.py nnet
python train.py dtree 
```

3) Evaluate the predictors

```bash
python eval.py -p nnet -m caldis
python eval.py -p nnet -m calplot
python eval.py -p nnet -m report
python eval.py -p nnet -m roc
python eval.py -p nnet -m confusion
python eval.py -p nnet -m misses














##Steps for setting model:
1) Set data file in data/sources.txt
2) Set ranges in data/ranges.txt
3) Set variables in utils/variables.txt

Inspecting missigness for current model:
```bash
python utils/missing.py
```

View scatterplot matrix for given dataset
```bash
python utils/scatterplot.py ./data/training-data-completed-9.csv
```


##Batch mode

1) Create training/test sets

```bash
python init.py -n [number of iterations] -t [test percentage] -i [number of imputed files to run MI on] -s [starting id]
```

```bash
python init.py -n 20 -s 0 -t 50 -m amelia [*]
```

2) Train the predictors

```bash
python train.py nnet
python train.py dtree
```

3) Evaluate the predictors

```bash
python eval.py -p nnet -m caldis
python eval.py -p nnet -m calplot
python eval.py -p nnet -m report
python eval.py -p nnet -m roc
python eval.py -p nnet -m confusion
python eval.py -p nnet -m misses
```


number of iterations = number of sets

test percentage = percentage of complete rows that will be used in the test sets

number of imputed files to run MI on = If you do -i 0 or just don't flag -i, it will delete the missing values.

starting id = will delete currents sets if is 0

2) 



###Score
Simple scoring scheme which counts how many variables fall outside the normal range 


###Neural Netork
Neural Network predictor trained on a dataset augmented with imputed data

0) Create variables.txt file in data dir with list of variables to use in the model. Each line in this file should contain the name of a variable (the short name, not the alias), followed by its Mirador type (separated by a space), which should be int, float, category, etc.

1) See to see summary of missingness with the currently selected variables:

```bash
python utils/missing.py
```

2) Create initial training and testing sets. The testing set is formed by a percentage of 
the complete rows, i.e.: those rows that don't have missing values for any of the selected
variables. The percentage is set to 75% by default, and it can be specified as an argument
for the script:

```bash
python utils/makesets.py 70
```

3) Create imputed datasets using Amelia and combine them into aggregated training dataset.
By default the number of imputed datasets is 10, but it can be specified as an argument
for the script:

```bash
python utils/impute.py 5
```

4) Generates a scatterplot matrix between all the independent variables in the model, 
labelled by the value of the dependent variable:

```bash
python utils/view.py
```

5) Train the neural network , and save optimal parameters to predictor.txt using the 
default parameters:

```bash
python nnet/train.py
```

The training algorithm has several parameters that can be edited inside the code.

6) Run predictor on test set, and calculate confusion matrix, accuracy, sensitivity and 
precision. Also prints the cases the predictor failed on:

```bash
python nnet/test.py
```

7) To generate a graphical representation of the neural network where use

```bash
python nnet/nnet.py
```

The grey nodes represent the bias terms, red edges correspond to negative weights, and
blue edges to positive edges. The width of the edges is proportional to the weight, 
normalized by the maximum weight within that layer.

8) Calculates and prints calibration and discrimination of predictive model.

```bash
python nnet/nnet-eval.py 1
```

9) Creates calibration plot and Hosmer-Lemeshow statistics.

```bash
python nnet/nnet-eval.py 2
```

10) Prints out a basic  classification report to evaluate predictive model. Includes
precision, recall, F1 score.

```bash
python nnet/nnet-eval.py 3
```

11) Computes and plots the ROC Curve from the predictive model.

```bash
python nnet/nnet-eval.py 4
```

###Decision Tree

0) Create variables.txt file in data dir with list of variables to use in the model. Each line in this file should contain the name of a variable (the short name, not the alias), followed by its Mirador type (separated by a space), which should be int, float, category, etc.

1) See to see summary of missingness with the currently selected variables:

```bash
python utils/missing.py
```

2) Create initial training and testing sets. The testing set is formed by a percentage of 
the complete rows, i.e.: those rows that don't have missing values for any of the selected
variables. The percentage is set to 75% by default, and it can be specified as an argument
for the script:

```bash
python utils/makesets.py 70
```

3) Create imputed datasets using Amelia and combine them into aggregated training dataset.
By default the number of imputed datasets is 10, but it can be specified as an argument
for the script:

```bash
python utils/impute.py 5
```

4) Generates a scatterplot matrix between all the independent variables in the model, 
labelled by the value of the dependent variable:

```bash
python utils/view.py
```

5) Train the decision tree , and save pickled model to data/dt-model.p:

```bash
python dt/dt_train.py
```

6) Run predictor on test set, and calculate confusion matrix, accuracy, sensitivity and 
precision. Also prints the cases the predictor failed on:

```bash
python dt/dt_test.py
```

7) To generate a graphical representation of the decision tree use

```bash
python dt/dt_draw.py
```

The grey nodes represent the bias terms, red edges correspond to negative weights, and
blue edges to positive edges. The width of the edges is proportional to the weight, 
normalized by the maximum weight within that layer.

8) Calculates and prints calibration and discrimination of predictive model.

```bash
python dt/dt_eval.py 1
```

9) Creates calibration plot and Hosmer-Lemeshow statistics.

```bash
python dt/dt_eval.py 2
```

10) Prints out a basic  classification report to evaluate predictive model. Includes
precision, recall, F1 score.

```bash
python dt/dt_eval.py 3
```

11) Computes and plots the ROC Curve from the predictive model.

```bash
python dt/dt_eval.py 4
```

###Comparing predictors

The relative performance of the predictors can be evaluated by running them several times and comparing
the averaged results:

```bash
python main.py -n [number of neural nets] -d [number of decision trees] -s [outfile for model] -e [evaluation method numbers ...]
```
You will need to pip install cloud.

All arguments are optional.
