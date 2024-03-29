# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:28:26 2024

@author: Mohamed AbdulHameed
"""
# This script reads in the training data, converts the SMILES strings to Morgan fingerprints 
# using the morgan_fingerprint function, stacks the fingerprint arrays into
#  a 2D numpy array X, defines a Naive Bayes classifier with the chosen 
# hyperparameter, and trains the classifier on all of the training data.

# Finally, the script saves the final model as both a pickle file and a joblib file.

# install these packages and then import 

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.naive_bayes import GaussianNB
import pickle
from joblib import dump

def morgan_fingerprint(smiles, radius, nBits):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp, dtype=np.int8)

# make sure label pre-processed csv file column name as 
# "ID", "SMILES", "CLASS"  

# Load dataset
data = pd.read_csv("hergi_all_preprocessed_out.csv")
df = data[["ID", "SMILES", "CLASS"]]

# Convert SMILES to Morgan fingerprints
df['Fingerprint'] = df['SMILES'].apply(morgan_fingerprint, radius=2, nBits=1024)

# Stack the fingerprint arrays into a 2D numpy array
X = np.stack(df['Fingerprint'].values)

# Split into input features and output variable
y = df['CLASS']

# Define the Naive Bayes classifier with the chosen hyperparameter
clf = GaussianNB(var_smoothing=1)

# Train the classifier on all of the training data
clf.fit(X, y)

# Save the model as a pickle file
with open('hERGi_NB_full_final_model.pickle', 'wb') as file:
    pickle.dump(clf, file)

# Train the final model on the entire training set
final_model = GaussianNB(var_smoothing=1)
final_model.fit(X, y)

# Save the final model using joblib
dump(final_model, 'hERGi_NB_full_final_model.joblib')
