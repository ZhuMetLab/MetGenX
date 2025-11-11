"""
# File       : Prepare_dataset.py
# Time       : 2025/11/3 14:17
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

from MS2Tools.SpectrumFileReader import SpectraData
import numpy as np
import pandas as pd
import os

project_dir = "./test/MassSpecGym"
metadata = pd.read_csv(os.path.join(project_dir, "MassSpecGym.tsv"), sep="\t")
spec_dict = { }
for idx, row in metadata.iterrows():
    mz = [float(m) for m in row["mzs"].split(",")]
    intensity = [float(m) for m in row["intensities"].split(",")]
    precursor_mz = row["precursor_mz"]
    name = row["identifier"]
    smiles = row["smiles"]
    inchikey = row["inchikey"]
    formula = row["formula"]
    adduct = row["adduct"]
    instrument = row["instrument_type"]
    metaData = {
        "id": name,
        "PEPMASS": precursor_mz,
        "smiles": smiles,
        "inchikey": inchikey,
        "formula": formula,
        "adduct": adduct,
        "instrument": instrument
    }
    spec = SpectraData(metaData, np.array(mz), np.array(intensity))
    spec_dict[name] = spec

from tqdm import tqdm
from  MS2Tools.SpectrumFileReader import export_mgf
for key, spec in tqdm(spec_dict.items()):
    export_mgf(spec,  key_map = None, file = "./test/MassSpecGym/MS2_spectra.mgf", append = True)
metadata.to_csv("./test/MassSpecGym/metaData.csv", index=False)