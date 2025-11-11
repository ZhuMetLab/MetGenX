"""
# File       : datasets.py
# Time       : 2025/10/23 10:17
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

import typing as T
import numpy as np
import torch
from Model.datasets.ProcessingFormula import generate_formula
from torch.utils.data.dataset import Dataset
def collater(tokens, pad_idx, pad_fixed_length=None):
    if pad_fixed_length is None:
        max_len = max(len(token) for token in tokens)
    else:
        if isinstance(pad_fixed_length, int):
            max_len = pad_fixed_length
        else:
            raise ValueError("The value of pad_fixed_length should be int.")
    token_size = len(tokens[0].size())
    if token_size==1:
        pad_tokens = [torch.cat((token, torch.tensor([pad_idx] * (max_len - len(token))))) if len(token) < max_len
                      else token[:max_len] for token in tokens]
    else:
        pad_tokens = [torch.cat((token, torch.tensor([pad_idx] * (max_len - len(token))).unsqueeze(1).repeat(1, token.size(1)))) if len(token) < max_len
                      else token[:max_len] for token in tokens]
    pad_tokens = torch.stack(pad_tokens)
    return pad_tokens

def token_smiles(smiles_list=None, tokenizer=None):
    token_list = tokenizer.tokenize_smiles(smiles_list)
    id_list = tokenizer.tokens_to_ids(token_list, pad=False)
    return id_list


from rdkit import Chem
def smiles_to_inchi_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey1 = Chem.MolToInchiKey(mol).split("-")[0]
    return inchikey1

class TemplateDataset(Dataset):
    def __init__(
            self,
            Template_dict: dict, DB_dict: dict,
            metadata,
            return_mol_freq: bool = True,
            return_identifier: bool = True,
            dtype: T.Type = torch.float32,
            tokenizer=None
    ):

        self.metadata = metadata
        self.inchikey_dict = dict(zip(metadata["identifier"], metadata["inchikey"]))
        self.return_mol_freq = return_mol_freq
        if self.return_mol_freq:
            if "inchikey" not in self.metadata.columns:
                self.metadata["inchikey"] = self.metadata["smiles"].apply(smiles_to_inchi_key)
            self.metadata["mol_freq"] = self.metadata.groupby("inchikey")["inchikey"].transform("count")

        self.return_identifier = return_identifier
        self.dtype = dtype

        self.bos_idx = 4
        self.pad_idx = 0
        self.pad_fixed_length = None

        self.Template_dict = Template_dict
        self.DB_dict = DB_dict
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.Template_dict)

    def __getitem__(
            self, i) -> dict:

        item = {}

        metadata = self.metadata.iloc[i]
        ID = metadata["identifier"]
        Template_list = self.Template_dict[ID]
        Template_list = Template_list[0:10]
        num_peaks = len(Template_list)
        item["num_peaks"] = torch.tensor([num_peaks])

        # Apply all transformations to the spectrum
        inchi1 = self.inchikey_dict[ID]
        item["formula"] = self.DB_dict[inchi1]["formula"]

        form_raw = generate_formula(self.DB_dict[inchi1]["formula"])
        score = [Template[1] for Template in Template_list]
        item["score"] = torch.tensor(score)

        cpd_id = [Template[0] for Template in Template_list]
        item["template_names"] = cpd_id

        FP_vec = torch.tensor(np.array([self.DB_dict[cpd]["fingerprint"] for cpd in cpd_id]))
        item["FP_vec"] = FP_vec

        form_vec = torch.stack([generate_formula(self.DB_dict[cpd]["formula"]) for cpd in cpd_id]).squeeze(1)
        item["form_vec"] = form_vec

        diff_form_vec = form_vec - form_raw
        item["diff_form_vec"] = diff_form_vec

        # SMILES
        Src_SMILES = self.DB_dict[inchi1].get("smiles")
        if Src_SMILES is not None:
            # mol_str
            item["mol"] = Src_SMILES
            Src_SMILES = token_smiles(smiles_list=[Src_SMILES], tokenizer=self.tokenizer)[0]
            dec_SMILES = Src_SMILES[:-1]
            dec_SMILES.insert(0, 4)
            item["Src_SMILES"] = torch.tensor(Src_SMILES).long()
            item["dec_SMILES"] = torch.tensor(dec_SMILES).long()
            # pesudo spec
            item["spec"] = item["Src_SMILES"]


        # Add other metadata to the item
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        # TODO: this should be refactored
        for k, v in item.items():
            if not isinstance(v, list):
                if not isinstance(v, str):
                    item[k] = torch.as_tensor(v, dtype=self.dtype)
        return item

    def collate_fn(self, samples):
        if len(samples) == 0:
            return {}
        batch = {}
        for k, v in samples[0].items():
            if isinstance(v, torch.Tensor):
                if len(v.size())!=0:
                    batch[k] = collater(
                        [sample[k] for sample in samples], self.pad_idx, self.pad_fixed_length)
                else:
                    batch[k] = torch.stack([sample[k] for sample in samples])
            else:
                batch[k] = [sample[k] for sample in samples]
        return batch

