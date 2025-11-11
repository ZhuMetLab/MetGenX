"""
# File       : Embedding_database.py
# Time       : 2025/10/23 9:37
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
from tqdm import tqdm
from MS2Tools.util import CalSpecVec
from TemplateSearch.Embedding import GenerateSpec2vec
import numpy as np
def Embedding_spectra(spec_data, gensim_model):
    Embeded_dict = {}
    for ID, spec in tqdm(spec_data.items()):
        spec.normalize(
            normalize_to=1,
            consider_precursor=True,
            precursor_mz=spec.metaData["mz"],
            mz_tol=10,
            res_define_at=0.0)
        spec.denoise(
            precursor_mz=spec.metaData["mz"],
            threshold_rel=0.01,
            threshold_abs=0,
            mz_tol=10,
            ms2_noise=3.0,
            sn_threshold=3.0,
            clear_noise=True,
            res_define_at=0.0,
        )
        precursor_mz = spec.metaData["mz"]
        spec_token, spec_intensity = GenerateSpec2vec(spec.mz, spec.intensity,
                                                      round(precursor_mz, 2)
                                                      , TopN=100, TopN_mz=None, min_int=0.01, NL_range=[0.5, 200],
                                                      min_frag=1)
        search_vector = CalSpecVec(spec_token, spec_intensity, gensim_model, normalized=False)
        search_vector = np.array([search_vector]).astype('float32')
        Embeded_dict[ID] = search_vector.flatten()
    return Embeded_dict

import faiss
def Create_idx(Embeded_dict, target_id):
    index = faiss.IndexFlatIP(512)
    vectors = np.stack([Embeded_dict[ID] for ID in target_id]).astype('float32')
    faiss.normalize_L2(vectors)
    index.add(vectors)
    Query_idx = {"index_ID":target_id, "index":index}
    return Query_idx