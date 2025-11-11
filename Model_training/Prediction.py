"""
# File       : Prediction.py
# Time       : 2025/10/23 13:08
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import torch
from Model.datasets.ProcessingFormula import generate_formula
from Model.datasets.data_utils import CleanSMILESList
from tqdm import tqdm
def pred(model, batch, Clean_formula=True):
    formula_list = batch["formula"]
    Formula_vec = torch.stack([generate_formula(formula) for formula in formula_list]).squeeze(0)
    Formula_vec = Formula_vec.to(model.device)
    with torch.no_grad():
        res, score = model.decode(batch, model.generation_config, Formula_vec)
    batch_size = res.size(0)
    smiles_list = []
    for batch_idx in range(batch_size):
        searched_ids = res[batch_idx, :, :]
        id_list = searched_ids.cpu().numpy().tolist()
        token_list = model.SMITokenizer.ids_to_tokens(id_list)
        smiles = model.SMITokenizer.tokens_to_string(token_list)
        smiles = [smi[0] for smi in smiles]
        smiles_list.append(smiles)
    mols_pred = smiles_list
    searched_res = [list(zip(mols_pred[i], score[i])) for i in range(len(mols_pred))]

    # filter formula
    if Clean_formula:
        cleaned_res = []
        mols_pred = []
        for i in range(len(searched_res)):
            res, num_vaild_smi_i, num_correct_formula_i = CleanSMILESList(searched_res[i],
                                                                          tgt_formula=batch["formula"][i],
                                                                          clean_formula=True, clean_duplicated=True)
            cleaned_res.append(res)
            if len(res) != 0:
                mols_pred.append([smi[0] for smi in res])
            else:
                mols_pred.append([smi[0] for smi in searched_res[i]])
        searched_res = cleaned_res

    return searched_res

def GenerateDataset(model, dataset, Clean_formula=True):
    DB_dict = dataset.dataset.DB_dict
    Generated_dict = {}
    for batch in tqdm(dataset, total=len(dataset)):
        batch = dataset.dataset.collate_fn([batch])
        if batch["identifier"][0] in Generated_dict.keys():
            continue
        search_res = pred(model, batch, Clean_formula=Clean_formula)
        search_res = search_res[0]
        if len(search_res) != 0:
            smiles_list, score_list = zip(*search_res)
            template_names = batch["template_names"][0]
            template_smiles = [DB_dict[template_name]["smiles"] for template_name in template_names]
            template_scores = batch["score"][0].numpy().tolist()
            Generated_dict[batch["identifier"][0]] = {"smiles": smiles_list, "score": score_list,
                                                      "template_smiles": template_smiles,
                                                      "template_scores": template_scores}
        else:
            Generated_dict[batch["identifier"][0]] = {"smiles": [], "score": [],
                                                      "template_smiles": None, "template_scores": None}

    return Generated_dict