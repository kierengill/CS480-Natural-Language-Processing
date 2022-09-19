Kieren Singh Gill
ksg7699

driver_train.ipynb - file to create training.feature file
driver_test.ipynb - file to create test.feature file
functions.py file - file with one helper function

The model scored 31632 out of 32853 tags correctly:
Precision: 81.90
Recall: 91.93
F1 Measure: 90.91

The model uses the following features:

POS = POS tag
word = word
forward_POS = forward POS 
forward_word = forward word
forward2_POS = 2nd forward POS 
forward2_word = 2nd forward word
prev_POS = previous POS
prev_word = previous word
prev_BIO = previous word BIO tag
prev2_POS = 2nd previous POS
prev2_word = 2nd previous word
capital_letter = if the first letter is capital
is_number = if it is a number

Using these combination of features gave me the highest F1 score.