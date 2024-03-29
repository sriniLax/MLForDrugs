# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:51:03 2024

@author: Mohamed AbdulHameed
"""


import pandas as pd
from rdkit import Chem
from chembl_structure_pipeline import standardizer

# load the data
df = pd.read_csv("herg_input.csv")
df.shape()

# set standardize function

def standardize_init(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        m_no_salts = standardizer.get_parent_mol(molecule)
        tostandarize = m_no_salts[0]
        std_mol = standardizer.standardize_mol(tostandarize)
        canonical_smiles = Chem.MolToSmiles(std_mol)
        return canonical_smiles
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(e)
        return None

# Standardize, remove salts, and canonicalize the SMILES column
df['SMILES'] = df['SMILES'].apply(standardize_init)
df.shape 

# Remove duplicates based on SMILES only
df2 = df.drop_duplicates(subset='SMILES')
df2 = df2.drop('is_duplicate', axis=1)
# Write the output to a new CSV file
df2.to_csv('hergi_all_preprocessed_out.csv', index=False)