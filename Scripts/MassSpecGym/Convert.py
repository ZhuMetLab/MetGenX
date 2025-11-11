"""
# File       : Convert.py
# Time       : 2025/10/25 17:05
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""


import numpy as np
import pandas as pd

from Model_training.datasets import TemplateDataset
from Model.MassSpecGym.datasets import TemplateDataset_retrival
def convert_to_retrieval_dataset(
    template_dataset: TemplateDataset,
    SMILES_dict: dict,
    candidate_dict_dir=None,
) -> TemplateDataset_retrival:
    retriveval_dataset = TemplateDataset_retrival(
        Template_dict=template_dataset.Template_dict,
        DB_dict=template_dataset.DB_dict,
        metadata=template_dataset.metadata,
        return_mol_freq=template_dataset.return_mol_freq,
        return_identifier=template_dataset.return_identifier,
        dtype=template_dataset.dtype,
        tokenizer=template_dataset.tokenizer,
        SMILES_dict=SMILES_dict,
        candidate_dict_dir=candidate_dict_dir
    )
    return retriveval_dataset


if __name__ == '__main__':
    # modify this for your custom dataset
    template_dataset = np.load("./results/MassSpecGym/input_dataset.dataset", allow_pickle=True)
    metaData = pd.read_csv("./test/MassSpecGym/metaData.csv")
    SMILES_dict = dict(zip(metaData["identifier"], metaData["smiles"]))
    retriveval_dataset = convert_to_retrieval_dataset(template_dataset,
        SMILES_dict,
        candidate_dict_dir="./test/MassSpecGym/MassSpecGym_retrieval_candidates_formula_canoical.json")

    import pickle
    with open("./results/MassSpecGym/input_dataset_retrieval.dataset", "wb") as f:
        pickle.dump(retriveval_dataset, f)