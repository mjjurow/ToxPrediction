# ToxPrediction
A tool to predict the likely toxicity endpoints of organic molecules

The code is deployed as a GCP cloud function here: https://www.matthewjurow.com/projects/molecular-toxicity-modeling

The data is taken from PubChem. The model accepts a SMILES string molecular representation as an input, and has been trained on 10,000 molecules. Each molecule has been decomposed into 209 features to provide predictions for 12 distinct toxicity endpoints. 

As deployed, the model is tuned to minimize false positives so no one throws out a valuable target moleclue because they (incorrectly) believe it to be toxic. I can change the model to emphasize different outcomes if anyone is interested in a different set of priorities. 
