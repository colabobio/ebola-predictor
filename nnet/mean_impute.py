'''
This script imputes the missing values using means.
It should not be used in practice! Just for dev purposes.
'''

import numpy as np
import pandas as pd

training_file = "training-data.csv"
outfile = "training-data-imputed.csv"

training = pd.DataFrame.from_csv(training_file)
training.reset_index(inplace=True)

mean_training = training.replace('?', '0')
mean_training = mean_training.applymap(lambda x:float(x))
means = mean_training.mean()

training.replace('?', means, inplace=True)
training.to_csv(outfile, index=False)
	
print "Done."    