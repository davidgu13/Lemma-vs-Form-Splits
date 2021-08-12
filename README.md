# Lemma-vs-Form-Splits

This project contains an implementation of an LSTM + Attention Sequence-to-sequence model for Inflection, a well-known task in the domain of computational morphology, 
applied on manipulated datasets of SIGMORPHON 2020, task 0. The manipulation is simple -- instead of just forbiding a sample to appear both at the train, dev and test set, 
we forbid *forms of the same lemma* to appear on the different sets. This helps us to obtain a clearer picture of the model's generalization abilities.


The repo consists of two parts -- the script for generating the lemma files, and the network trained on the old and new datasets.