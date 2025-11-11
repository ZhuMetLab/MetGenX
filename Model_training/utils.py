"""
# File       : utils.py
# Time       : 2025/10/23 13:32
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

from rdkit import Chem
def GetStructureK(smiles_list, tgt_smiles):
    inchi = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(tgt_smiles))[0:14]
    inchi_list = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smi))[0:14] for smi in smiles_list]
    if inchi in inchi_list:
        return inchi_list.index(inchi) + 1
    else:
        return "NF"


import pandas as pd
def generate_performance_list(rank_list):
    hits = []
    hits_perc = []
    total = len(rank_list)
    Res_Found = [int(rank) for rank in rank_list if rank != "NF"]
    ranks = range(1, 11)
    for i in ranks:
        n_hit = len([x for x in Res_Found if x <= i])
        hits.append(n_hit)
        hits_perc.append(round(n_hit / total, 4))
    n_hit = len(Res_Found)
    hits.append(n_hit)
    hits_perc.append(round(n_hit / total, 4))
    index = list(ranks) + [11]
    Res_df = pd.DataFrame({"Hits": hits, "Percentage": hits_perc}, index=index)
    return Res_df

import os
def ConvertDataset():
    delta_mass_dict = {
        "[M+H]+": 1.0073,
        "[M+Na]+": 22.9893,
        "[M-H2O+H]+": -17.0033,
        "[M+H3N+H]+": 18.0339,
        '[M-H4O2+H]+': -35.0139,
        '[M]+': -0.0005,
        '[M+K]+': 38.9632
    }

    dataset_dir = "./Tidy_training_script/NPLIB1"
    metadata = pd.read_csv(os.path.join(dataset_dir, "metaData.csv"), sep="\t")
    labels = pd.read_csv(os.path.join("./data/NPLIB1", "canopus_train_export", "labels.tsv"), sep="\t")
    merged_labels = pd.merge(metadata, labels, how="left", left_on="ID", right_on="spec")
    metaData_msg_style = pd.DataFrame(
        {
            "identifier": merged_labels["ID"],
            "smiles": merged_labels["smiles"],
            "inchikey": merged_labels["inchikey1"],
            "formula": merged_labels["formula_x"],
            "parent_mass": merged_labels["monoisotopic_mass"],
            "precursor_mz": merged_labels["monoisotopic_mass"],
            "adduct": merged_labels["ionization"],
            "instrument_type": merged_labels["instrument"],
            "collision_energy": None,
            "fold": None
        }
    )
    precursor_mz = []
    for i, row in metaData_msg_style.iterrows():
        monoisotopic_mass = row.parent_mass
        adduct = row.adduct
        precursor_mz.append(monoisotopic_mass+ delta_mass_dict[adduct])
    metaData_msg_style["precursor_mz"] = precursor_mz
    split_file = os.path.join(dataset_dir,"canopus_hplus_100_0.tsv")
    split = pd.read_csv(split_file, sep="\t")
    training_id = split[split["split"] == "train"]["name"].tolist()
    valid_id = split[split["split"] == "val"]["name"].tolist()
    testing_id = split[split["split"] == "test"]["name"].tolist()
    for name in metaData_msg_style["identifier"]:
        if name in training_id:
            metaData_msg_style.loc[metaData_msg_style["identifier"] == name, "fold"] = "train"
        elif name in valid_id:
            metaData_msg_style.loc[metaData_msg_style["identifier"] == name, "fold"] = "val"
        elif name in testing_id:
            metaData_msg_style.loc[metaData_msg_style["identifier"] == name, "fold"] = "test"

    metaData_msg_style.to_csv(os.path.join(dataset_dir,"metaData.csv"), index=False)