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
predictor on the test set. The pipeline includes a number of built-in predictors (Logistic 
Regression, Decision Tree, and Neural Network classifiers), but additional predictors can 
be added to it by following a few coding conventions. 

Because missing values is a typical problem in survey and health data, the pipeline allows 
to "complete" an incomplete training set by using a variety of methods. Some of them are 
already built into the pipeline, but additional methods can be implemented also by following
a number of conventions.



###Step by step examaples

**1) SINGLE TRAINING/TESTING RUN**

create training/testing sets, impute missing values, train predictor,
evaluate predictor:

```
python utils/makesets.py
```

The percentage of complete records used in testing set can be specified with the -p argument:

```
python utils/makesets.py -p 70
```

Then, we need to impute the missing values. The options are: 

* list-wise deletion: removes a record if it contains any number of missing values. If there
are no missing values, it effectively acts as a copy operation.

* mean imputation: it calculates the mean value in each variable separately, and replaces 
the missing values with the means

* amelia: hybrid EM with bootstrap

* Hmisc: chained-equation using predicted mean matching or regression imputation

* MICE: chained-equation (and potential NINR models)

list-wise and mean imputation are not recommended, however list-wise should be used when
there are no missing values since it operates as a copy:

```
python utils/listdel.py
```

Amelia, Hmisc and MICE are run in a similar way:

```
python utils/amelia.py
python utils/mice.py
python utils/hmisc.py
```

Amelia, hmisc, and MICE have optional arguments that are different between them, but all 
allow to specify the number of imputed data frames that are combined to generate a
single training set:

```
python utils/amelia.py --num_imputed 10
python utils/mice.py --num_imputed 10
python utils/hmisc.py --num_imputed 10
```

This number is 5 by default.

After imputing missing values, the model training phase is executed by running the train 
script available in each predictor:

```
python nnet/train.py
python lreg/train.py
python scikit_lreg/train.py
python scikit_dtree/train.py
python scikit_randf/train.py
python scikit_svm/train.py
```
Each predictor has unique options, calling the train script with the -h argument will show
all the options.

```
python nnet/train.py -h
python nnet/train.py -r 100
```

Finally, evaluation is conducted with the eval script available for each predictor. The 
evaluation metric needs to be specified with the -m argument, which accepts the following
options: caldis, calplot, report, roc, confusion, misses:

```
python nnet/eval.py -m report
```

**2) BATCH MODE**

It is often the case we need to get a sense of the overall performance of the predictor 
when generating many different copies of training/testing sets. This can be achieved by
the provided batch mode. The first step in the batch mode consists in generating a 
predetermined number of training/testing sets, and completing the training sets with the
imputation algorithm of choice. For instance, to generate 10 training/testing sets with 
amelia imputation, one would run:

```
python init.py -n 10 -s 0 -m amelia
```

The percentage of complete records to include in the testing set, as well as imputation 
arguments can be specified as well:

```
python init.py -n 10 -s 0 -t 70 -m amelia num_imputed=10
```

The training of the desired predictor on each of the generated training sets can be performed 
with:

```
python train.py nnet
```

One can pass the arguments to the predictor as follows:

```
python train.py nnet inv_reg=100
```

The evaluation over all testing sets and summary statistics can be obtained with the eval script:

```
python eval.py -p nnet -m report
```

**2.1) BATCH MODE WITH CUSTOM MODEL NAME and LOCATION**

The batch mode scripts allow to set the name of the model being trained/tested, as well
as the location where the generated files will be stored. By model, here it is understood
the collection of predictive variables set in the data/variables.txt file. These parameters
are specified with the -N and -B arguments in the init, train, and eval scripts:

```
python init.py -B /Users/andres/Temp/test -N simple -n 10 -s 0 -m amelia
python train.py -B /Users/andres/Temp/test -N simple nnet
python eval.py -B /Users/andres/Temp/test -N simple -p nnet -m report
```

**3) GENERATING/RUNNING JOBS**

Exhaustive model generation can take a long time on a single machine, so a job generation/
launching system is provided to run these tasks on HPC clusters.

As a preparation step, clean all previous job files:

```
python clean.py -j
```

The first step consists in adding all variables in data/variables-master.txt. If one
variable is to be kept constant, the star character can be added at the end of the line for
that variable.

Then, run the gen_jobs.py script with the sizes of models to use:

```
python gen_jobs.py -s 2-5 -c 5
```

The range of model sizes is is given in the argument -s, while -c indicates how many
models to "clump" together in each job.

```
python run_jobs.py -m lsf
```

In order to see how the job submission commands would look like, but without actually submitting 
them, we can use the debug mode:

```
python run_jobs.py -m debug
```

And to run them locally

```
python run_jobs.py -m local
```

Each job will operate on the corresponding variables generated with gen_jobs, and will
use the configuration set in the job.cfg file

To get a ranking of all the models and predictors for each model, the rank_models script
will go through each folder and parse the report files and sort the predictions based on 
their F1-scores:

```
python rank_models.py
```

As earlier, a custom location can be used to store the models and output files. This needs to 
be set in the job.cfg file so each job will save its data to the corresponding model folder,
as well as an argument to run_jobs and rank_models:

```
python run_jobs.py -B /Users/andres/Temp/test/ -m debug
python rank_models.py -B /Users/andres/Temp/test/
```



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

The makesets script also makes sure that the same proportion of output classes are present 
in the training and test sets.

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
by default it will generate 5 imputed datasets, which are then aggregated to create an 
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
three predictors: the [Decision Tree classifier](http://scikit-learn.org/stable/modules/tree.html) 
from [scikit-learn](http://scikit-learn.org/), the [Logistic Regression classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 
also from [scikit-learn](http://scikit-learn.org/), and a standard single-layer Neural Network. 
All are trained in a similar way. When using the default options, it is enough to do:

```bash
python dtree/train.py 
```
for the Decision Tree,

```bash
python lreg/train.py 
```

for the Logistic Regression, and

```bash
python nnet/train.py 
```

for the Neural Network. In all cases, the training set is expected to be located in *data/training-data-completed.csv*.
The parameters of the trained Logistic Regression, Decision Tree and Neural Network predictors 
are saved by default to *data/lreg-params*, *data/dtree-params*, and *data/nnet-params*, respectively.

Each predictor has several additional arguments. The Logistic Regression classifier:

```bash
python lreg/train.py [-h] [-t TRAIN] [-p PARAM] [-y PENALTY] [-d DUAL] [-c INV_REG]
                     [-f FIT_INTERCEPT] [-s INTERCEPT_SCALING] [-w CLASS_WEIGHT]
                     [-r RANDOM_STATE] [-l TOL]
```

In the case of the Decision Tree:

```bash
python dtree/train.py [-h] [-t TRAIN] [-p PARAM] [-c CRITERION] [-s SPLITTER]
                      [-maxf MAX_FEATURES] [-maxd MAX_DEPTH]
                      [-mins MIN_SAMPLES_SPLIT] [-minl MIN_SAMPLES_LEAF]
                      [-maxl MAX_LEAF_NODES]
```

In both cases the name of the training and parameter files can be customized with the 
-t and -p arguments, while the rest of the arguments affecting the Logistic Regression and 
the Decision Tree algorithms are documented in the corresponding scikit-learn documentation 
pages, [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 
and [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

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
there are several measures that can be used to quantify its performance. The Logistic 
Regression, Decision Tree and the Neural Network predictors provide evaluation scripts:

```bash
python lreg/eval.py
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

In all cases when the evaluation method generates a plot (calplot, roc), they are saved 
in pdf format in the *out* folder.

###Plotting

In addition to the plots generated by some of the evaluation methods (calplot, roc) and the
missingness plots from Amelia, the pipeline provides some additional graphical outputs to 
visualize the training and test sets and the parameters of the Decision Tree and Neural 
Network predictors.

* Scatterplot matrix of a data file:

```bash
python utils/scatterplot.py [-h] [data]
```

* Graph of Neural Network given parameters file:

```bash
python nnet/view.py [-h] [param]
```

* Graph of Decision Tree given parameters file:

```bash
python dtree/view.py [-h] [param]
```

###Dependencies

The pipeline has been tested on Python 2.7.5. The basic dependencies are the following Python
packages:

* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [matplotlib](http://matplotlib.org/)
* [scikit-learn](http://scikit-learn.org/)

Additional dependencies:

* The amelia script and the calplot evaluation method require [R](http://www.r-project.org/) 
and [RPy2](https://pypi.python.org/pypi/rpy2), and the [Amelia](http://cran.r-project.org/web/packages/Amelia/index.html) 
and [PredictABEL](http://cran.r-project.org/web/packages/PredictABEL/index.html) packages for R, respectively.
* The view script in the Decision Tree requires the python package [pydot](https://pypi.python.org/pypi/pydot), 
as well as [GraphViz](http://www.graphviz.org/) installed in the system.

One way to install all dependencies is to use one of the Python scientific software collections that are now available.
The pipeline has been tested with [Anaconda](https://store.continuum.io/cshop/anaconda/) from [Continuum Analytics](http://continuum.io/).

###Batch Mode

Sometimes it could be useful to re-train the predictor several times, evaluate the performance
for each training round, and finally present an average evaluation. This is possible using 
the batch mode scripts: *init*, *train*, and *eval*. The *init* script generates as many 
training/test set pairs as specified, and applies a data completion method on the training set.
The *train* script will train a specific predictor using all the available training sets, and
will save the resulting parameters into separate files. Finally, the *eval* script runs one
evaluation method on all the parameters available for the selected predictor, and calculates
the average results. All these scripts are separate so costly steps (data imputation, model 
training) need to be run only once.

**1) Initialization.** This step involves generating the training/test sets, and completing 
the training data:

```bash
python init.py [-h] [-n NUMBER] [-s START] [-t TEST] 
               [-m METHOD] [data completion arguments]
```

* -n, --number: number of training/test pairs to generate
* -s, --start: starting id of first training/test pair. This can be useful when a previous
  init run was interrupted midway through, and we need to pick up where it stopped in order
  to reach a certain number of sets. When starting from zero, all existing training/test sets
  are removed.
* -t, --test: percentage of complete rows to build the test sets
* -m, --method: data completion method, must be the name of the scrip implementing the method
  (without the .py extension)
* data completion arguments: any argument that the data completion script is able to accept

For example, if we want to generate 20 training/test sets, starting from zero, using 60% of 
the complete rows for the test sets, and imputing missing values in the training data using 
10 imputed datasets from Amelia, then we would do:

```bash
python init.py -n 20 -s 0 -t 60 -m amelia num_imputed=10
```

All the Amelia arguments desribed earlier can be added to the init script using the name=value
syntax.

**2) Training.** The train script will train the selected predictor on each training file
generated in the previous step:

```bash
python train.py [-h] pred [predictor arguments]
```

* pred: name of the predictor to train (lreg for Logistic Regression, dtree for Decision Tree, 
nnet for Neural Network)
* predictor arguments: any argument accepted by the predictor, in the format name=value

For instance, in order to train the Decision Tree using the entropy criterion and 5 as 
max_depth we need:

```bash
python train.py dtree criterion=entropy max_depth=5
```

**3) Evaluation.** The evaluation step will run the selected evaluation method for all the
predictor parameters generated during training, using the corresponding test sets:

```bash
python eval.py [-h] [-p P] [-m M]
```
* -p, --predictor: name of the predictor to evaluate (lreg for Logistic Regression, dtree for Decision Tree, 
nnet for Neural Network)
* -m, --method: evaluation method. Must be one from the following: caldis, calplot, report, 
  roc, confusion, misses (default report)

###Packing/unpacking models

All the information that defines a predictive model, together with the training/test sets
and generated parameters are stored inside the *data* folder, so a few utility additional
scripts are provided to ease cleaning, backing up, and restoring its contents.

* Cleaning data: Use *clean.py* script to remove any generated files in *data* and restore it
to its original state:

```bash
python clean.py
```

* Packing data: The data folder can be packed into a zip file inside the *store* folder 
with the *pack.py* script:

```bash
python pack.py [name of zip file]
```

* Unpacking data: A zip file generated with pack can be restored into the *data* folder 
using *unpack.py*. All previous files in *data* are removed:

```bash
python unpack.py [name of zip file]
```

###Reference prediction: EPS

The EPS (Ebola Prognosis Score) is provided as a predictor, although it does not require 
training. It is included to provide a baseline reference to all predictors trained on the 
[Ebola dataset](http://fathom.info/mirador/ebola/datarelease).

EPS is defined on two sets of variables, one comprising only PCR and clinical symptoms 
(EPS7), and another including lab results (EPS10). The files containing these sets are
*data/variables-eps7.txt* and *data/variables-eps10.txt*, which need to be copied into 
*data/variables.txt* prior to any analysis using EPS.

Since no training is needed, all the (complete) data can be allocated to the test set:

```bash
python utils/makesets.py -p 100
```

The evaluation script works similarly to the ones available for the built-in predictors:

```bash
python eps/eval.py --cutoff 0 --method report
```

but it has a cutoff argument, which sets the EPS score at or below which the outcome prediction 
is survival.

The *stats.py* script is also provided in the EPS folder to calculate some basic statistics:

* A [T-test](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) 
to determine if the mean score for surviving and deceased Ebola patients is significantly different.

* A [Fisher-exact](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html) 
test on the a 2x2 contingency table that results from cross tabulating observed outcome with
a calculated EPS "prediction" defined as survival if the score is < score mean + score std,
die otherwise.

The script is run as

```bash
stats.py [-h] [-t TEST] [-o OUT] [-c CUTOFF] [-n NAME]
```

with:

* -t, --test: name of test file containing the cases
* -o, --out: name of csv file to save the cases and the components of the EPS score
* -n, --name: name of the EPS being used
* -c, -cutoff: it can be used as the cutoff of the EPS prediction, overriding of mean + std.

###Neural Network with PCR test

The nnet-pcr module is provided as a customized version of the Neural Network predictor,
which adds an additional step to the prediction after evaluating the probabilities. This
test consists in switching the prediction from 0 to 1 when the PCR value is higher than
a threshold set in the code of eval.py. This module doesn't implement the training stage,
since it is identical to the default Neural Network. It only provides the evaluation 
methods that operate on the modified predictor, which uses the available nnet parameters.

##Advanced: implementing custom modules

New data imputation and machine learning algorithms can be added to the pipeline by 
providing the corresponding scripts. These scripts need to follow a few coding conventions
so they can be integrated into the pipeline properly.

###Custom data imputation scripts

They need to be placed inside the *utils* folder. They requirement is that they need to have
a *process(in_filename, out_filename)* method which at least accepts two arguments: the name
of the input file holding the original training set, and the name of the output file where
the completed dataset will be stored:

```python
# Template for data imputation script

import argparse

def process(in_filename, out_filename):
    print "Imputing missing data in",in_filename
    print "Saving imputed data to",in_filename
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=1, default=["./data/training-data.csv"],
                        help="name of input training file")
    parser.add_argument('-o', '--output', nargs=1, default=["./data/training-data-completed.csv"],
                        help="name of completed training file")
    args = parser.parse_args()
    process(args.input[0], args.output[0])
```

###Custom predictors

New predictors need to have their own folder, and at least two scripts inside their folder, 
train.py and eval.py, which should implement the training and testing stages. Templates for
these scripts are provided below:

```python
# Template for predictor training script

import argparse

# Predictor prefix
def prefix():
    return "pred"

# Predictor full name
def title():
    return "Predictor"

def train(train_filename, param_filename):
    print "Training..."

if __name__ == "__main__":
    parser.add_argument("-t", "--train", nargs=1, default=["./data/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./data/pred-params"],
                        help="File contaning predictor parameters")
    args = parser.parse_args()
    train(parser.train[0], parser.param[0])
```

```python
# Template for predictor evaluation script

import os, argparse
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix, run_eval, get_misses

# Predictor prefix
def prefix():
    return "pred"

# Predictor full name
def title():
    return "Predictor"

def pred(test_filename, train_filename, param_filename):
    X, y = design_matrix(test_filename, train_filename)

    # Need to compute the probability of output = 1 for each row of the design matrix:
    probs = []
    for i in range(0, len(X)):
        probs.extend(random.random() for x in X[i,1:]]) # Dummy calculation

    return probs, y

# Prints and returns the output of the evaluation
def eval(test_filename, train_filename, param_filename, method, **kwparams):
    X, y = design_matrix(test_filename, train_filename)
    
    # Need to compute the probability of output = 1 for each row of the design matrix:
    probs = []
    for i in range(0, len(X)):
        probs.extend(random.random() for x in X[i,1:]]) # Dummy calculation
    
    return run_eval(probs, y, method, **kwparams)
    
# Prints miss-classifications, returns indices
def miss(test_filename, train_filename, param_filename):
    fn = test_filename.replace("-data", "-index")
    meta = None
    if os.path.exists(fn):
        with open(fn, "r") as idxfile:
            meta = idxfile.readlines()

    X, y, df = design_matrix(test_filename, train_filename, get_df=True)
    
    # Need to compute the probability of output = 1 for each row of the design matrix:
    probs = []
    for i in range(0, len(X)):
        probs.extend(random.random() for x in X[i,1:]]) # Dummy calculation
    
    indices = get_misses(probs, y)
    for i in indices:
        print "----------------"
        if meta: print "META:",",".join(lines[i].split(",")).strip()
        print df.ix[i]
    return indices

def evaluate(test_filename, train_filename, param_filename, method):
    # Average calibrations and discriminations
    if method == "caldis":
        eval(test_filename, train_filename, param_filename, 1)
    # Plot each method on same calibration plot
    elif method == "calplot":
        eval(test_filename, train_filename, param_filename, 2, test_file=test_filename)
    # Average precision, recall, and F1 scores
    elif method == "report":
        eval(test_filename, train_filename, param_filename, 3)
    # Plot each method on same ROC plot
    elif method == "roc":
        eval(test_filename, train_filename, param_filename, 4, pltshow=True)
    # Average confusion matrix
    elif method == "confusion":
        eval(test_filename, train_filename, param_filename, 5)
    # Method not defined:
    elif method == "misses":
        miss(test_filename, train_filename, param_filename)
    else:
        raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs=1, default=["./data/training-data-completed.csv"],
                        help="Filename for training set")
    parser.add_argument('-T', '--test', nargs=1, default=["./data/testing-data.csv"],
                        help="Filename for testing set")
    parser.add_argument('-p', '--param', nargs=1, default=["./data/pred-params"],
                        help="Filename for predictor parameters")
    parser.add_argument('-m', '--method', nargs=1, default=["report"],
                        help="Evaluation method: caldis, calplot, report, roc, confusion, misses")
    args = parser.parse_args()
    evaluate(args.test[0], args.train[0], args.param[0], args.method[0])
```







