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











##Dependencies
* Pandas
* Numpy
* matplotlib
* scikit-learn

* RPy2
* R (Amelia II, PredictABEL)
* pydot, Graphviz










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
