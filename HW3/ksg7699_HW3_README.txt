Kieren Gill
ksg7699
NLP HW3

To run the program, run the file titled "ksg7699_HW3_driver.ipynb".
It is a Jupyter notebook file that creates the training set using "WSJ_02-21.pos" and the test set using "WSJ_23.words".
Also included is a functions.py file, which contains all the functions used in the .ipynb notebook.

Unfortunately, I was not able to handle OOV items well. I tried tagging them with different unknown tags in the assign_unk function
(--unk-- , --unk_digit--, --unk_punct--, etc.) but I wasn't sure how to incorporate them effectively into my program. 

Functions for processing/organizing data:
    #build_vocab
    #build_vocab2idx
    #processing
    #training_data
    #assign_unk
    #get_word_tag
    #create_dictionaries

Functions for calculations:
    #create_transition_matrix
    #create_emission_matrix
    #initialize_transition_matrix
    #vterbi_forward
    #vterbi_backward

I was able to get a 6/12 on the autograder, with 47609 out of 56684 tags correct.