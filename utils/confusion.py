'''Create a confusion matrix.'''

def confusion(probs, y_test):
        
    n_hit = 0          # Hit or True Positive (TP)
    n_correct_rej = 0  # Correct rejection or True Negative (TN)
    n_miss = 0         # Miss or False Negative (FN)    
    n_false_alarm = 0  # False alarm, or False Positive (FP)
    
    for i in range(len(probs)):
        p = probs[i]
        pred = 0.5 < p
        
        if pred == 1:
            if y_test[i] == 1: 
                n_hit = n_hit + 1
            else: 
                n_false_alarm = n_false_alarm + 1
        else:
            if y_test[i] == 1: 
                n_miss = n_miss + 1
            else: 
                n_correct_rej = n_correct_rej + 1
        
    return (n_hit, n_false_alarm, n_miss, n_correct_rej)
    
   