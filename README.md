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

The pipeline roughly involves four stages: defining predictive model, constructing training 
and test sets, fitting the desired predictor using the training set, and evaluating the 
predictor on the test set. The pipeline includes a number of built-in predictors (a Decision 
Tree and a Neural Network), but additional predictors can be added to it by following a few 
coding conventions. 

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
python utils/amelia.py [-h] [-i INPUT] [-o OUTPUT] 
                       [-n NUM_IMPUTED] [-r NUM_RESAMPLES] [-c] [-p]
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

**3) Training the predictor.** Once a completed training set is constructed in the previous 
step, the desired Machine Learning predictor can be trained. The pipeline currently includes
two predictors: the [Decision Tree classifier](http://scikit-learn.org/stable/modules/tree.html) 
from [scikit-learn](http://scikit-learn.org/), and a custom Neural Network. Both are trained 
in a similar way. When using the default options, it is enough to do:

```bash
python dtree/train.py 
```
for the Decision Tree, and

```bash
python nnet/train.py 
```

for the Neural Network. In both cases, the training set is expected to be located in *data/training-data-completed.csv*.
The parameters of the trained Decision Tree and Neural Network predictors are saved by default 
to *data/dtree-params* and *data/nnet-params*, respectively.

Each predictor has several additional arguments. In the case of the Decision Tree:

```bash
python dtree/train.py [-h] [-t TRAIN] [-p PARAM] [-c CRITERION] [-s SPLITTER]
                      [-maxf MAX_FEATURES] [-maxd MAX_DEPTH]
                      [-mins MIN_SAMPLES_SPLIT] [-minl MIN_SAMPLES_LEAF]
                      [-maxl MAX_LEAF_NODES]
```

The name of the training and parameter files can be customized with the -t and -p arguments, 
while the rest of the arguments affect the Decision Tree algorithm itself, and are 
documented in the scikit-learn documentation page for the [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

The Neural Network allows to tweak several elements of the algorithm as well:

```bash
python nnet/train.py [-h] [-t TRAIN] [-p PARAM] [-l LAYERS] [-f HFACTOR] [-g GAMMA]
                     [-c CONVERGENCE] [-s] [-d]
```                

* -l , --layers: represents the number of hidden layers (default 1)
* -f, --hfactor: it is the factor used to calculate the number of nodes per hidden layer. 
  The calculation is hfactor * number of input variables (default 1)
* -g, --gamma: is the regularization coefficient, which penalizes large values of the
  Neural Network coefficients. This can help to reduce overfitting (default 0.002)
* -c, --convergence: is the threshold to stop the gradient minimization step in the 
  coefficient optimization (default 1E-5)
* -s, --show: shows the plot of the scores along the coefficient optimization path
* -d, --debug: debugs the gradient calculation by comparing the result of the analytical formula
  with the numerical estimation

**4) Testing the predictor.** A trained predictor is evaluated on the testing set, however
there are several measures that can be used to quantify its performance. Both the Decision 
Tree and the Neural Network provide evaluation scripts:

```bash
python dtree/eval.py
python nnet/eval.py
```

which by default will use the default training and test set filenames from previous steps. 
The available options are the same in both cases:

```bash
eval.py [-h] [-t TRAIN] [-T TEST] [-p PARAM] [-m METHOD]
```

* -t, --train: name of training file (it is needed to determine the value ranges used for 
  normalization in the training step)
* -T, --test: name of testing file
* -p, --param: name of parameters file
* -m, --method: evaluation method. Must be one from the following: caldis, calplot, report, 
  roc, confusion, misses (default report)

The evaluation methods are briefly described below:

* caldis: [calibration and discrimination](http://blog.yhathq.com/posts/predicting-customer-churn-with-sklearn.html#calibration-and-descrimination) 
  calculations. Calibration quantifies how far is the frequency of positive outcomes from 
  the actual probability of those outcomes (so lower
  values are better), while the discrimination measures how far from the base probability 
  is the true frequency each output category (so higher values are better).  
* calplot: calibration plot and [Hosmer-Lemeshow test statistic](http://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test), 
  as computed by [preditABEL](http://www.genabel.org/PredictABEL/plotCalibration.html). 
  These measures allow to compare observed and predicted risks.
* report: [precision](http://en.wikipedia.org/wiki/Precision_and_recall), 
  [recall](http://en.wikipedia.org/wiki/Precision_and_recall) 
  (also called sensitivity) and [F1-scores](http://en.wikipedia.org/wiki/F1_score). These
  are standard measures in Machine Learning to characterize the performance of a binary
  classifier. A high precision means that the predictor yields few false 
  positives, while a high recall or sensitivity indicates that the is able to return most
  of positive outcomes (few false negatives). The F1-score is a weighted average of the 
  precision and recall.
* roc: [receiver operating characteristic curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic),
  represents the ratio of the [sensitivity](http://en.wikipedia.org/wiki/Sensitivity_and_specificity) 
  to the fall-out ([1-specificity](http://en.wikipedia.org/wiki/Sensitivity_and_specificity)), for
  different discrimination thresholds.
* confusion: [confusion matrix](http://en.wikipedia.org/wiki/Confusion_matrix), showing the
  counts of true positives, true negatives, false positives, and false negatives.
* misses: list of miss-classified elements (false positives, and false negatives) in the test set.

In all cases when the evaluation generates a plot, they are saved in pdf format in the *out* folder.

###Plotting
```bash
python utils/scatterplot.py ./data/training-data-completed-9.csv
```

###Dependencies
* Pandas
* Numpy
* matplotlib
* scikit-learn

* RPy2
* R (Amelia II, PredictABEL)
* pydot, Graphviz




###Bath Mode

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

###Packing/unpacking models



##Reference prediction: EPS

Use either variables-eps7 or variables-eps10, set ranges to all ages then:

```bash
python utils/makesets.py -p 100
```

```bash
python eps/eval.py --cutoff 0 --method report
```
cutoff sets the EPS score above which a patient is predicted to die.

















