''' Performs optional initialization

It can be absent from the code
'''
def init():
    pass

''' Returns the list of variables needed to compute the new variable

The names must correspond to the original titles in the data file
'''
def variables():
    return ["VOMIT", "WEAK", "EDEMA", "CONF"]

''' Returns the name of the new variable

The names must correspond to the original titles in the data file
'''
def get_name():
    return "CLS"

''' Returns the title of the new variable

The title is often a human-readable string describing the variable
'''
def get_title():
    return "Clinical Score"

''' Returns a string with the data type for the variable

The valid types are int, long, float, double, and category
'''
def get_type():
    return "float"

''' Returns composite value given the values of the component variables

The returned value will be automatically converted into a string, so it might be a good
idea to do the conversion directly inside this method to make sure it is correct.
'''
def calculate(values):
    all_missing = True
    score = 0
    
    for var in ["VOMIT", "WEAK", "EDEMA", "CONF"]:
        val = values[var]
        if val != "\\N":
            all_missing = False
            score += float(val)
        
    if all_missing: return "\\N"
    
    print values, score
    return score
