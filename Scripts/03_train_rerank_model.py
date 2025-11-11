"""
# File       : 03_train_rerank_model.py
# Time       : 2025/10/23 13:06
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
from Model.BART.BART import BARTModel
from Model.Configs import config
from Model.datasets.Vocabulary import Vocabulary
from Model.datasets.Tokenizer import SMILESTokenizer
import os
import torch
from torch.utils.data import Subset
from Model_training.Prediction import GenerateDataset
import pickle
from Model_training.utils import generate_performance_list

# change the dataset name
dataset_name = "NPLIB1" # name of your dataset

# Training the rerank model
# load data
path_train = os.path.join("results", dataset_name, "input_dataset.dataset")
dataset = np.load(path_train, allow_pickle=True)
config_path = os.path.join("results", dataset_name, "./weights/generation/config.json")

print("Loading config ...")
try:
    model_config = config.Config.from_json_file(config_path)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")

vocab = Vocabulary(special_tokens=["<bos>", "<eos>"])
vocab_path = os.path.join("results", dataset_name, "./weights/generation/vocab.txt")
vocab.Load_vocab(vocab_path)
Convert_path = os.path.join("results", dataset_name, "./weights/generation/Convert_dict.dict")
Convert_dict = np.load(Convert_path, allow_pickle=True)
tokenizer = SMILESTokenizer(vocab=vocab, hydrate=True, Convert_dict=Convert_dict)
config_generation = os.path.join("results", dataset_name, "./weights/generation/config_generation.json")
generate_config = config.Config_generation.from_json_file(config_generation)
generate_config.vocab = vocab
model = BARTModel(
    model_config=model_config,
    use_pretrained=False,
    use_formula=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

checkpoint = torch.load(os.path.join("results", dataset_name, "./weights/generation/Trained_Weight.ckpt"), map_location=device)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# Generate SMILES for training
# example: using the validation set
split = dataset.metadata[["identifier", "fold"]]
split = split.set_index("identifier")["fold"]
split_mask = split.loc[dataset.metadata["identifier"]].values
val_dataset = Subset(dataset, np.where(split_mask == "val")[0])

model.generation_config = generate_config
model.SMITokenizer = tokenizer
Generated_dict = GenerateDataset(model, val_dataset, Clean_formula=True)
with open(os.path.join("results", dataset_name, "./weights/generation/Validation_generation_result.pkl"), "wb") as f:
    pickle.dump(Generated_dict, f)

# train reranking model
from Model_training.RerankModel import train, Rerank_data
metaData = dataset.metadata
SMILES_dict = dict(zip(metaData["identifier"], metaData["smiles"]))
if not os.path.exists(os.path.join("results", dataset_name, "./weights/rerank")):
    os.makedirs(os.path.join("results", dataset_name, "./weights/rerank"))
gbm = train(save_dir=os.path.join("results", dataset_name, "./weights/rerank/"), Generated_dict=Generated_dict
      , SMILES_dict=SMILES_dict)


import lightgbm as lgb
# example: Generating results in the test set
gbm = lgb.Booster(model_file=os.path.join("results", dataset_name, "./weights/rerank/checkpoint_gbm"))
test_dataset = Subset(dataset, np.where(split_mask == "test")[0])
Generated_dict = GenerateDataset(model, test_dataset, Clean_formula=True)
with open(os.path.join("results", dataset_name, "./weights/generation/Testing_generation_result.pkl"), "wb") as f:
    pickle.dump(Generated_dict, f)
Testing_data_rerank = Rerank_data(Generated_dict, SMILES_dict, gbm)
Testing_data_rerank.to_csv(os.path.join("results", dataset_name, "Testing_rerank_generation_result.csv"))
Res_df = generate_performance_list(Testing_data_rerank["rank"])
for top in [1, 3, 5, 10]:
    if top in Res_df.index:
        print(f"Top{top} Accuracy: {Res_df.loc[top, 'Percentage']:.4f}")
