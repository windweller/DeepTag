# DeepTag
This contains the source code for the DeepTag Project

Unfortunately due to data share agreement, we are not at liberty to share our data.

However, we are able to share our experiment framework code. Note the randomness in CuDNN (an algorithm framework that we used) cannot be completely fixed, and the training process is stochastic. Replicating our number exactly is difficult, but we have run each experiment 5 times and report the average number.

`csu_model.py` is a very long code that has `Trainer`, `Experiment` set up. `Trainer` manages `Classifier`, it trains and loads classifiers. `Experiment` manages the entire experiment, the folder structure, automatically recording each random run, compute average statistics automatically. This code is very modular, and can be used in many ML researches.

`csu_data.py` preprocesses the data.

`Learning_to_Reject.ipynb` is the Jupyter Notebook that we used to compute the abstention algorithm. It loads in previously saved output from a trained classifier, and use it to train a new abstention model. The plots in the paper are generated from this notebook.

`data/snomed_label_to_meta_grouping.json` contains the label (disease) similarity that we defined. We hope this list to be of general value to people working with SNOMED-CT disease level codes.
