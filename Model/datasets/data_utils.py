# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch


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

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
def CleanSMILESList(searched_res, tgt_smiles=None, tgt_formula=None, clean_formula=True, clean_duplicated=True):
    new_res = []
    new_score = []
    new_inchi = []
    num_vaild_smi = 0
    num_correct_formula = 0
    for smi2, score in searched_res:
        try:
            mol2 = Chem.MolFromSmiles(smi2)
            formula2 = CalcMolFormula(mol2)
            canonical_smi2 = Chem.inchi.MolToInchiKey(mol2)[0:14]
            num_vaild_smi = num_vaild_smi + 1
            if clean_formula:
                if tgt_smiles is not None:
                    mol1 = Chem.MolFromSmiles(tgt_smiles)
                    tgt_formula = CalcMolFormula(mol1)
                elif tgt_formula is None:
                    raise ValueError("Both tgt_smiles and tgt_formula are None.")
                if formula2 != tgt_formula:
                    continue
                else:
                    num_correct_formula = num_correct_formula + 1
            if clean_duplicated:
                if canonical_smi2 not in new_inchi:
                    new_inchi.append(canonical_smi2)
                else:
                    continue
            new_res.append(smi2)
            new_score.append(score)
        except:
            pass
    return list(zip(new_res, new_score)), num_vaild_smi, num_correct_formula




def get_free_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    gpu_memory_usage = []

    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_mem = total_mem - free_mem
        gpu_memory_usage.append((i, used_mem))
    best_gpu = min(gpu_memory_usage, key=lambda x: x[1])[0]

    print(f"Using GPU: {best_gpu}")
    return f"cuda:{best_gpu}"

