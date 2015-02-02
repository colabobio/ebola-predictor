##Ebola Outcome Predictors

Algorithms to predict outcome of Ebola patients given their clinical and lab symptoms. Based on the metadata available at http://fathom.info/mirador/ebola/datarelease.html

Steps for setting model:
1) Set data file in data/sources.txt
2) Set ranges in data/ranges.txt
3) Set variables in utils/variables.txt



Batch mode
1) Create training/test sets
```bash
python utils/create_sets.py -n [number of iterations] -t [test percentage] -i [number of imputed files to run MI on*] -s [starting id]
```
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
