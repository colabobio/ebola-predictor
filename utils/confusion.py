"""
Creates a confusion matrix.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

label_file = "./data/labels.txt"

def confusion(probs, y_test):
    target_names = []
    with open(label_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            target_names.append(line.split()[1])

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

    print "Confusion matrix"
    print "{:25s} {:20s} {:20s}".format("", "Output " + target_names[1], "Output " + target_names[0])
    print "{:25s}{:2.0f}{:19s}{:2.0f}".format("Predicted " + target_names[1], n_hit,"", n_false_alarm)
    print "{:25s}{:2.0f}{:19s}{:2.0f}".format("Predicted " + target_names[0], n_miss,"", n_correct_rej)

    return (n_hit, n_false_alarm, n_miss, n_correct_rej)
