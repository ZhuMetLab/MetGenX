

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gensim
import numpy as np
from TemplateSearch.Embedding import GenerateSpec2vec
from MS2Tools.util import CalSpecVec
import faiss
from rdkit import Chem
class QueryTemplates(object):
    def __init__(self, Embed_dict:dict=None, index_dict:dict=None, d_model=512, SMILES_dict=None,
                 gensim_model_dir=None):
        if index_dict is None:
            index, indexlist = self.__BuildIndex(Embed_dict, d_model=d_model, normalize=True)
        else:
            index = index_dict["index"]
            indexlist = index_dict["index_ID"]
        self.index = index
        self.indexlist = indexlist
        # self.modification_vec = np.load("./model/table/modification_vec.pkl",allow_pickle=True)
        if not gensim_model_dir:
            gensim_model_dir = "weights/word2vec/SpecEmbed_model"
        self.gensim_model = gensim.models.Word2Vec.load(gensim_model_dir)
        self.smiles_dict = SMILES_dict

    def __BuildIndex(self,Embed_dict, d_model=512, normalize=True):
        index = faiss.IndexFlatIP(d_model)
        vectors = np.stack([Embed_dict[ID] for ID in Embed_dict.keys()]).astype('float32')
        if normalize:
            faiss.normalize_L2(vectors)
        index.add(vectors)
        indexlist = list(Embed_dict.keys())
        return index, indexlist

    def Embed_MS2(self, spec):
        intensity = spec.intensity.tolist()
        intensity = [peak / max(intensity) for peak in
                     intensity]
        spec_token, spec_intensity = GenerateSpec2vec(spec.mz.tolist(), intensity,
                                                      round(spec.metaData['mz'], 2)
                                                      , TopN=100, TopN_mz=None, min_int=0.01, NL_range=[0.5, 200],
                                                      min_frag=1)
        search_vector = CalSpecVec(spec_token, spec_intensity, self.gensim_model, normalized=False)
        search_vector = np.array([search_vector]).astype('float32')
        return search_vector

    def find_templates(self, formula, search_vector, DB_cutoff=0.5, use_modification=True, exclude_smiles=None):
        search_dict = self.search_similarity(search_vector=search_vector, normalize=True, cutoff=DB_cutoff)
        template_list = [(self.smiles_dict[template], search_dict[template], template) for template in
                  search_dict.keys()]
        if exclude_smiles is not None:
            inchi1_list = [Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[0:14] for smi in exclude_smiles]
            templates_filter = []
            for template in template_list:
                inchi1 = Chem.MolToInchiKey(Chem.MolFromSmiles(template[0]))[0:14]
                if inchi1 not in inchi1_list:
                    templates_filter.append(template)

                if len(templates_filter)>=10:
                    break
        else:
            templates_filter = template_list[0:10]

        return templates_filter

    def CalSimilarity(self, search_vector, normalize=True):
        index = self.index
        NSearched = len(self.indexlist)
        if len(search_vector.shape) == 1:
            search_vector = np.array([search_vector]).astype('float32')
        if normalize:
            faiss.normalize_L2(search_vector)
        distance, indexes = index.search(search_vector, NSearched)
        return distance, indexes

    def search_similarity(self, search_vector, normalize=True, cutoff=0.5):
        distance, indexes = self.CalSimilarity(search_vector, normalize)
        SearchedSpec = [self.indexlist[idx] for idx in list(indexes[0])]
        distances = list(distance[0])
        search_dict = dict(zip(SearchedSpec, distances))
        search_dict = {k: v for k, v in search_dict.items() if v >= cutoff}
        return search_dict

