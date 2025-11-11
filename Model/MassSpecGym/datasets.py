"""
# File       : datasets.py
# Time       : 2025/10/24 9:36
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import typing as T
import numpy as np
import torch
from Model.datasets.ProcessingFormula import generate_formula
from torch.utils.data.dataset import Dataset
from massspecgym.data.transforms import MolToInChIKey, MolFingerprinter
import json

def token_smiles(smiles_list=None, tokenizer=None):
    token_list = tokenizer.tokenize_smiles(smiles_list)
    id_list = tokenizer.tokens_to_ids(token_list, pad=False)
    return id_list

from rdkit import Chem
def smiles_to_inchi_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey1 = Chem.MolToInchiKey(mol).split("-")[0]
    return inchikey1

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


class TemplateDataset_retrival(Dataset):
    def __init__(
            self,
            Template_dict: dict, DB_dict: dict,
            metadata,
            return_mol_freq: bool = True,
            return_identifier: bool = True,
            dtype: T.Type = torch.float32,
            tokenizer=None, SMILES_dict=None,
            mol_label_transform  = MolToInChIKey(),
            mol_transform = MolFingerprinter(),
            candidate_dict_dir = "./data/MassSpecGym_retrieval_candidates_formula_canoical_filter.json"
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

        self.SMILES_dict = SMILES_dict
        self.mol_label_transform = mol_label_transform
        self.mol_transform = mol_transform

        with open(candidate_dict_dir, "r", encoding="utf-8") as f:
            candidate_dict = json.load(f)


        self.candidates = candidate_dict

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


        # retrival
        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = self.SMILES_dict[ID]

        # Get candidates
        if item["smiles"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["smiles"]}.')
        item["candidates"] = self.candidates[item["smiles"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]


        # Create neg/pos label mask by matching the query molecule with the candidates
        item_label = self.mol_label_transform(item["smiles"])
        item["labels"] = [
            self.mol_label_transform(c) == item_label for c in item["candidates"]
        ]
        # labels = item["labels"]
        # if labels.count(True) > 1:
        #     print(ID)
        #     first_true_idx = labels.index(True)
        #     mask = [i == first_true_idx or not lbl for i, lbl in enumerate(labels)]
        #     item["candidates"] = [c for c, keep in zip(item["candidates"], mask) if keep]
        #     item["candidates_smiles"] = [c for c, keep in zip(item["candidates_smiles"], mask) if keep]
        #     item["labels"] = [lbl for lbl, keep in zip(labels, mask) if keep]


        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        # Transform the query and candidate molecules
        item["mol"] = self.mol_transform(item["mol"])
        item["candidates"] = [self.mol_transform(c) for c in item["candidates"]]
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)


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
            if k not in ["candidates", "labels", "candidates_smiles"]:
                if isinstance(v, torch.Tensor):
                    if len(v.size())!=0:
                        batch[k] = collater(
                            [sample[k] for sample in samples], self.pad_idx, self.pad_fixed_length)
                    else:
                        batch[k] = torch.stack([sample[k] for sample in samples])
                else:
                    batch[k] = [sample[k] for sample in samples]

        # Collate candidates and labels by concatenating and storing sizes of each list
        batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in samples])
        )
        batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in samples], start=[])
        )
        batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in samples]
        )
        batch["candidates_smiles"] = sum([
            item["candidates_smiles"] for item in samples],start=[])
        return batch
