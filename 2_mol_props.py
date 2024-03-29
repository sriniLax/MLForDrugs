# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:06:24 2024

@author: Mohamed AbdulHameed
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# Load the data
data = pd.read_csv("hergi_all_preprocessed_out.csv")

# Define a function to calculate the molecular properties
def calc_mol_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    logP = Descriptors.MolLogP(mol)
    MW = Descriptors.MolWt(mol)
    rotB = Descriptors.NumRotatableBonds(mol)
    HBA = Descriptors.NumHAcceptors(mol)
    HBD = Descriptors.NumHDonors(mol)
    nRING = Descriptors.RingCount(mol)
    TPSA = Descriptors.TPSA(mol)
    
    return pd.Series([logP, MW, rotB, HBA, HBD, nRING, TPSA])

# Calculate molecular properties for each compound
data[["logP", "MW", "rotB", "HBA", "HBD", "nRING", "TPSA"]] = data["SMILES"].apply(calc_mol_properties)

# Write out molecular properties to generate boxplots
data.to_csv("hergi_all_7molrpop.csv", index=False)