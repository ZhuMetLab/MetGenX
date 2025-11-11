"""
# File       : 01_Construct_dataset.py
# Time       : 2025/10/23 8:54
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import os
import pandas as pd
from TemplateSearch.Embedding import GenerateSpec2vec
from MS2Tools.SpectrumFileReader import SpectrumFileReader
from Model_training.Train_embedding_model import train_gensim_model
import gensim
from Model_training.Embedding_database import Embedding_spectra, Create_idx
from Model_training.Build_dataset import build_similarity_dict
import pickle
import numpy as np
from Model.datasets.smiles import SMILESStandarder
from Model.Fingerprints.fingerprinting import Fingerprinter
from Model_training.Build_dataset import Create_datasets
from tqdm import tqdm

### Modify the following parameters before running

dataset_dir = "./test/MassSpecGym"  # path of your dataset
dataset_name = "MassSpecGym" # name of your dataset
save_dir = os.path.join("./results", dataset_name) # path to save results
split_file = None # path of split file (.tsv)
sampled_K = 10000 # Maximum Top-K templates for each query (necessary to save time and memory for larger spectra datasets)
spectra_cutoff = 0.4 # cutoff for spectra similarity score for template search

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# load data
spec_data = SpectrumFileReader(os.path.join(dataset_dir, "MS2_spectra.mgf")).read_file()
metadata = pd.read_csv(os.path.join(dataset_dir, "metaData.csv"))
spec_dict = {os.path.splitext(spec.metaData["id"])[0]: spec for spec in spec_data} # the key in MS2 spectra is id

###

# training embedding model
if not os.path.exists(os.path.join(save_dir, "weights", "word2vec")):
    os.makedirs(os.path.join(save_dir, "weights", "word2vec"))

# vectorization
if not os.path.exists(os.path.join(save_dir, "weights", "word2vec", "SpecEmbed_model")):
    Training_mz = []

    for ID, spec in tqdm(spec_dict.items()):
        mz = spec.mz
        intensity = spec.intensity
        precursor_mz = spec.metaData["mz"]
        spec_token, spec_intensity = GenerateSpec2vec(mz, intensity,
                                                      round(precursor_mz, 2)
                                                      , TopN=100, TopN_mz=None, min_int=0.01, NL_range=[0.5, 200],
                                                      min_frag=1)
        Training_mz.append(spec_token)
    model = train_gensim_model(Training_mz, model_save_dir=os.path.join(save_dir, "weights", "word2vec"))

# embedding
gensim_model = gensim.models.word2vec.Word2Vec.load(os.path.join(save_dir, "./weights/word2vec/SpecEmbed_model"))
Embeded_dict = Embedding_spectra(spec_dict, gensim_model)
if not os.path.exists(os.path.join(save_dir, "Query_index")):
    os.makedirs(os.path.join(save_dir, "Query_index"))

with open(os.path.join(save_dir,"./Query_index/embedding_spectra.db"), "wb") as f:
    pickle.dump(Embeded_dict, f)

# dataset_split
if split_file is not None:
    split = pd.read_csv(split_file, sep="\t")
    training_id = split[split["split"] == "train"]["name"].tolist()
    valid_id = split[split["split"] == "val"]["name"].tolist()
    testing_id = split[split["split"] == "test"]["name"].tolist()
    for name in metadata["identifier"]:
        if name in training_id:
            metadata.loc[metadata["identifier"] == name, "fold"] = "train"
        elif name in valid_id:
            metadata.loc[metadata["identifier"] == name, "fold"] = "val"
        elif name in testing_id:
            metadata.loc[metadata["identifier"] == name, "fold"] = "test"

else:
    if "fold" in metadata.columns:
        training_id = metadata[metadata["fold"] == "train"]["identifier"].tolist()
        valid_id = metadata[metadata["fold"] == "val"]["identifier"].tolist()
        testing_id = metadata[metadata["fold"] == "test"]["identifier"].tolist()
    else:
        raise ValueError("No split information provided")
        # training_id = list(Embeded_dict.keys())
        # valid_id = []
        # testing_id = []

# ensure all ids in Embeded_dict
training_id = [x for x in training_id if x in Embeded_dict.keys()]
valid_id = [x for x in valid_id if x in Embeded_dict.keys()]
testing_id = [x for x in testing_id if x in Embeded_dict.keys()]

# Create Query index
# Embeded_dict_train = {k:v for k,v in Embeded_dict.items() if k in training_id}
# with open(os.path.join(save_dir,"./Query_index/embedding_spectra_training.db"), "wb") as f:
#     pickle.dump(Embeded_dict_train, f)

if not os.path.exists(
        os.path.join(save_dir, "Query_index", "Dataset_training.idx")):
    Query_idx = Create_idx(Embeded_dict, target_id=training_id)
    with open(os.path.join(save_dir, "Query_index", "Dataset_training.idx"), "wb") as f:
        pickle.dump(Query_idx, f)
else:
    Query_idx = np.load(os.path.join(save_dir, "Query_index", "Dataset_training.idx"), allow_pickle=True)

if not os.path.exists(os.path.join(save_dir,"temp")):
    os.makedirs(os.path.join(save_dir,"temp"))

# Search Templates
if not os.path.exists(os.path.join(save_dir, "temp", "Template_dict_training.pkl")):
    for split_name, ids in zip(
        ["training", "validation", "test"],
        [training_id, valid_id, testing_id]
    ):
        template_dict = build_similarity_dict(ids, Embeded_dict, Query_idx, K=sampled_K)
        save_path = os.path.join(save_dir, "temp", f"Template_dict_{split_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(template_dict, f)


# Calculate smiles and fingerprints
if not os.path.exists(os.path.join(save_dir, "temp", "DB_dict.pkl")):
    compound_set = metadata.drop_duplicates(subset='inchikey')
    standarder = SMILESStandarder()
    smiles_canoical = [standarder.Standard(smi, isomericSmiles=False, canonical=True, kekuleSmiles=True) for smi in
                       tqdm(compound_set["smiles"])]
    Input_df = pd.DataFrame(
        {"ID": compound_set["identifier"], "formula": compound_set["formula"],"monoisotopic_mass":compound_set["parent_mass"], "SMILES": smiles_canoical, "inchikey1": compound_set['inchikey'] })
    # Fingerprint
    Fingerprinter_test = Fingerprinter(
        lib_path="Model/Fingerprints/fingerprint-wrapper/target/fingerprint-wrapper-bin-0.5.2.jar")
    res = Fingerprinter_test.process_df(Input_df, in_column="SMILES")
    fingerid_df = pd.read_table(os.path.join("./weights/generation", "Fingerprints.tsv"))
    used_index = list(fingerid_df.absoluteIndex)
    res["FP_used"] = [FP[0][used_index] for FP in res["fingerprint"]]
    with open(os.path.join(save_dir, "temp", "DB_dict.pkl"), "wb") as f:
        pickle.dump(res, f)
else:
    res = np.load(os.path.join(save_dir, "temp", "DB_dict.pkl"), allow_pickle=True)

# Create dataset
inchikey_list = list(res["inchikey1"])
smiles_list = list(res["SMILES"])
formula_list = list(res["formula"])
FP_list = list(res["FP_used"])
DB_dict = {inchikey_list[i]: {"smiles": smiles_list[i], "fingerprint": FP_list[i],
                              "formula": formula_list[i]} for i in range(len(inchikey_list))}
with open(os.path.join(save_dir, "temp", "DB_dict.dict"), "wb") as f:
    pickle.dump(DB_dict, f)


Template_dict_list = []
for ids in ["training", "validation", "test"]:
    Template_dict_list.append(np.load(os.path.join(save_dir, "temp", f"Template_dict_{ids}.pkl"), allow_pickle=True))

input_dataset = Create_datasets(Template_dict_list, metadata, DB_dict, template_num=10, cutoff=spectra_cutoff)

with open(os.path.join(save_dir, "input_dataset.dataset"), "wb") as f:
    pickle.dump(input_dataset, f)


import shutil
from pathlib import Path
weights_dir = "./weights"
copy_file_list = [
    "./generation/config.json",
    "./generation/config_database.json",
    "./generation/Convert_dict.dict",
    "./generation/vocab.txt",
    "./generation/config_generation.json",
    "./generation/Fingerprints.tsv",
    "./res_dict.cfg"
]

target_folder = os.path.join(save_dir)
Path(target_folder).mkdir(parents=True, exist_ok=True)

for src_path_str in copy_file_list:
    src_path = Path(os.path.join(weights_dir, src_path_str))
    if not src_path.exists():
        print(f"File not found: {src_path_str}")
        continue
    relative_path = src_path.relative_to(".")
    dst_path = Path(target_folder) / relative_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"Copied {src_path} -> {dst_path}")