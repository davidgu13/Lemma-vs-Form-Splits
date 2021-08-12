# Lemma-vs-Form-Splits

This project contains an implementation of an LSTM + Attention Sequence-to-sequence model for Inflection, a well-known task in the domain of computational morphology, 
applied on manipulated datasets of SIGMORPHON 2020, task 0. The manipulation is simple -- instead of just prohibiting *a sample* (a *lemma,form,features* triplet) to appear both at the train, dev and test set, we prohibit samples with *forms of the same lemma* to appear on the different sets. This helps us to obtain a clearer picture of the model's generalization abilities.


The repo consists of the script that generates the lemma files and the network trained on the old and new datasets, implemented with PyTorch.

To generate the new datasets, clone the original SIGMORPHON data and and run generate_lemma_splits.py in the same folder.

To train the network, in Inflection_90_Langs.py select the list langauges to train on and run the script.
