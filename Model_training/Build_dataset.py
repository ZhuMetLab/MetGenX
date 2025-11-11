"""
# File       : Build_dataset.py
# Time       : 2025/10/23 9:50
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
from tqdm import tqdm
import faiss
import numpy as np

def build_similarity_dict(query_id, Embeded_dict, Query_idx, K=10000):
    """
    {id1: [(id2, score), (id3, score), ...]}
    """
    similarity_dict = {}
    index_ID = Query_idx["index_ID"]
    index = Query_idx["index"]

    for i, id1 in tqdm(enumerate(query_id), total=len(query_id)):
        search_vector = np.array([Embeded_dict[id1]], dtype="float32")
        faiss.normalize_L2(search_vector)
        distances, indexes = index.search(search_vector, K)  # Top-K

        neighbors = []
        for dist, idx in zip(distances[0], indexes[0]):
            if idx < 0:
                continue
            neighbor_id = index_ID[idx]
            if neighbor_id == id1:
                continue
            neighbors.append((neighbor_id, float(dist)))

        similarity_dict[id1] = neighbors

    return similarity_dict


from Model.datasets.Tokenizer import SMILESTokenizer
from Model.datasets.Vocabulary import Vocabulary
from Model_training.datasets import TemplateDataset
def Create_datasets(Template_dict_list, metadata, DB_dict, template_num=10, cutoff=0.4):
    vocab = Vocabulary(special_tokens=["<bos>", "<eos>"])
    vocab_path = "./weights/generation/vocab.txt"
    vocab.Load_vocab(vocab_path)
    Convert_path = "./weights/generation/Convert_dict.dict"
    Convert_dict = np.load(Convert_path, allow_pickle=True)
    tokenizer = SMILESTokenizer(vocab=vocab, hydrate=True, Convert_dict=Convert_dict)
    InChiKey_list = dict(zip(metadata["identifier"], metadata["inchikey"]))
    Template_dict_combine = {}
    for Template_dict in Template_dict_list:
        Template_dict_filtered = {}
        for key, value in tqdm(Template_dict.items()):
            InChIKey_pool = []
            item_pool = []
            for item in value:
                target_InChikey = InChiKey_list[item[0]]
                if target_InChikey == InChiKey_list[key]:
                    continue
                if target_InChikey not in InChIKey_pool:
                    InChIKey_pool.append(target_InChikey)
                    item_pool.append((target_InChikey, item[1]))
            Template_dict_filtered[key] = item_pool
        Template_dict_combine.update(Template_dict_filtered)

    Template_dict_filtered = {}
    for key, values in tqdm(Template_dict_combine.items()):
        value = [value for value in values if value[1]>=cutoff]
        if len(value)>0:
            Template_dict_filtered[key] = value[0:10]
    Template_dict_combine = Template_dict_filtered
    Template_dict_combine = {ID: template[0:template_num] for ID, template in Template_dict_combine.items() if len(template) > 0}
    metadata_filter = metadata[metadata["identifier"].isin(Template_dict_combine.keys())]
    input_dataset = TemplateDataset(Template_dict=Template_dict_combine, DB_dict=DB_dict,
                                    metadata=metadata_filter, tokenizer=tokenizer)

    return input_dataset