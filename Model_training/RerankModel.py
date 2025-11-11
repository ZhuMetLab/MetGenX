"""
# File       : RerankModel.py
# Time       : 2025/10/23 13:36
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import os

import numpy as np
from Rerank.features import Feature_calculater
import lightgbm as lgb
from tqdm import tqdm

remain_features = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

params = {
    'task': 'train',
    'boosting_type': 'gbrt',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'metric_freq': 1,
    'train_metric': True,
    'ndcg_at': [10],
    'max_bin': 255,
    'num_iterations': 1000,
    'learning_rate': 0.01,
    'num_leaves': 11,
    'tree_learner': 'serial',
    'min_data_in_leaf': 30,
    'verbose': 2
}

def split_list(input_list, seed, split_ratio=0.3):
    np.random.seed(seed)
    np.random.shuffle(input_list)
    split_index = int(len(input_list) * split_ratio)
    list1 = input_list[:split_index]
    list2 = input_list[split_index:]
    return list1, list2

def data_prepare(Generated_dict, SMILES_dict, save_dir, split_data=False, split_ratio=0.3, seed=42):
    Calculater = Feature_calculater(Feature_cache=None, Cal_extra_FP=True, CanonicalTautomer=True,
                                    FP_type="Morgan")
    if split_data:
        training_id, validation_id = split_list(list(Generated_dict.keys()), seed=seed, split_ratio=split_ratio)
        Generated_dict_train = {key: Generated_dict[key] for key in training_id}
        Generated_dict_val = {key: Generated_dict[key] for key in validation_id}
        process_dict = {"training":Generated_dict_train, "validation":Generated_dict_val}
    else:
        process_dict = {"training":Generated_dict}

    for datasetname, Generated_dict in process_dict.items():
        groups = []
        for ID, data in tqdm(Generated_dict.items()):
            if data["template_smiles"] is None:
                continue
            template_used = list(zip(data["template_smiles"], data["template_scores"]))
            candidates = list(zip(data["smiles"], data["score"]))
            tgt_smiles = SMILES_dict[ID]
            n_templates = len(template_used)
            n_candidates = len(candidates)
            Features = Calculater.Combine_features(candidates, template_used, n_templates, n_candidates,
                                                   remain_features=remain_features)
            labels = Calculater.Cal_labels(candidates, tgt_smiles, FP_type="Daylight")
            lines = []
            for i in range(len(Features)):
                rows = f"{round(labels[i], 4)} " + " ".join([f"{i + 1}:{val}" for i, val in enumerate(Features[i])])
                lines.append(rows)
            if len(Features) != 0:
                groups.append(len(Features))
            with open(f'{save_dir}/{datasetname}_data.txt', 'a') as f:
                for line in lines:
                    f.write(line + '\n')
        with open(f'{save_dir}/{datasetname}_groups.txt', 'a') as f:
            for line in groups:
                f.write(str(line) + '\n')

from sklearn import datasets
def train(save_dir,skip_data_prepare=False, Generated_dict=None, SMILES_dict=None, split_data=False, split_ratio=0.3, seed=42):
    if not skip_data_prepare:
        data_prepare(Generated_dict, SMILES_dict, save_dir, split_data=split_data, split_ratio=split_ratio, seed=seed)
    else:
        if not os.path.exists(f'{save_dir}/training_data.txt') or not os.path.exists(f'{save_dir}/training_groups.txt'):
            raise ValueError(f'{save_dir}/training_data.txt or {save_dir}/training_groups.txt does not exist')

    feats = f'{save_dir}/training_data.txt'
    group = f'{save_dir}/training_groups.txt'
    x_train, y_train = datasets.load_svmlight_file(feats)
    y_train = (y_train * 10).astype(int)
    q_train = np.loadtxt(group)
    train_data = lgb.Dataset(x_train, label=y_train, group=q_train)

    if os.path.exists(f'{save_dir}/validation_data.txt') and not os.path.exists(f'{save_dir}/validation_groups.txt'):
        feats = f'{save_dir}/validation_data.txt'
        group = f'{save_dir}/validation_groups.txt'
        x_test, y_test = datasets.load_svmlight_file(feats)
        y_test = (y_test * 10).astype(int)
        q_test = np.loadtxt(group)
        test_data = lgb.Dataset(x_test, label=y_test, group=q_test)
        gbm = lgb.train(params, train_data, valid_sets=[test_data])
    else:
        gbm = lgb.train(params, train_data)
    gbm.save_model(os.path.join(save_dir, "checkpoint_gbm"))
    return gbm

from Model_training.utils import GetStructureK
import pandas as pd
def Rerank_data(Generated_dict, SMILES_dict, gbm):
    Calculater = Feature_calculater(Feature_cache=None, Cal_extra_FP=True, CanonicalTautomer=True,
                                    FP_type="Morgan")
    reranked_smiles = []
    modified_scores = []
    modified_tanimoto = []
    modified_rank = []
    raw_score = []
    raw_rank = []
    for ID, res in tqdm(Generated_dict.items()):
        if res["template_smiles"] is None:
            raw_score.append([])
            reranked_smiles.append([])
            modified_scores.append([])
            modified_tanimoto.append([])
            modified_rank.append("NF")
            tgt_smiles = SMILES_dict[ID]
            raw_rank.append(GetStructureK(res["smiles"], tgt_smiles))
            continue
        template_used = list(zip(res["template_smiles"], res["template_scores"]))
        candidates = list(zip(res["smiles"], res["score"]))
        tgt_smiles = SMILES_dict[ID]
        n_templates = len(template_used)
        n_candidates = len(candidates)
        if len(candidates) != 0:
            Features = Calculater.Combine_features(candidates, template_used, n_templates, n_candidates,
                                                   remain_features=remain_features)
            labels = Calculater.Cal_labels(candidates, tgt_smiles, FP_type="Daylight")
            data = pd.DataFrame(Features)
            modified_score = gbm.predict(data)
            sorted_indexes = np.argsort(modified_score)[::-1]
            sorted_smiles = [candidates[i][0] for i in sorted_indexes]
            sorted_Tanimoto = [labels[i] for i in sorted_indexes]
            sorted_score = [modified_score[i] for i in sorted_indexes]
            rank = GetStructureK(sorted_smiles, tgt_smiles)
            sorted_raw_score = [candidates[i][1] for i in sorted_indexes]
            raw_rank.append(GetStructureK(res["smiles"], tgt_smiles))
            raw_score.append(sorted_raw_score)
            reranked_smiles.append(sorted_smiles)
            modified_scores.append(sorted_score)
            modified_tanimoto.append(sorted_Tanimoto)
            modified_rank.append(rank)

    df_rerank = pd.DataFrame({
        "ID": list(Generated_dict.keys()),
        "rank": modified_rank,
        "raw_rank": raw_rank,
        "tgt_smi": [SMILES_dict[ID] for ID in Generated_dict.keys()],
        "pred_smiles": reranked_smiles,
        "Top1_tanimoto": [score[0] if len(score) != 0 else None for score in modified_tanimoto],
        "Tanimoto": modified_tanimoto,
        "score": modified_scores,
        "raw_score": raw_score
    })
    return df_rerank
