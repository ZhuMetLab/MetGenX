"""
# File       : Generation.py
# Description:
"""

from Generation.pred_spec import Generator
from TemplateSearch.QueryDB import QueryTemplates
from Model.Configs.config import Config_databse
import numpy as np
import pandas as pd
import faiss


def get_meta(meta_dict, key):
    for k, v in meta_dict.items():
        if k.lower() == key.lower():
            return v
    return None


class MetGenX(object):
    def __init__(self,
                 polarity="positive",
                 config_path="./weights/generation/config.json",
                 config_generation_path="./weights/generation/config_generation.json",
                 config_database_path="./weights/generation/config_database.json"
                 ):

        Config_db = Config_databse.from_json_file(config_database_path)
        if polarity == "Nodb":
            db = None
        elif polarity == "positive":
            MetInfo = np.load(Config_db.MetInfo_pos, allow_pickle=True)
            embed_dict = np.load(Config_db.spec_pos, allow_pickle=True)
            SMILES_dict = dict(zip(MetInfo["ID"], MetInfo["SMILES"]))
            gensim_model_dir = Config_db.embed_pos
            db = QueryTemplates(embed_dict, d_model=512, SMILES_dict=SMILES_dict, gensim_model_dir=gensim_model_dir)
        else:
            MetInfo = np.load(Config_db.MetInfo_neg, allow_pickle=True)
            embed_dict = np.load(Config_db.spec_neg, allow_pickle=True)
            SMILES_dict = dict(zip(MetInfo["ID"], MetInfo["SMILES"]))
            gensim_model_dir = Config_db.embed_neg
            db = QueryTemplates(embed_dict, d_model=512, SMILES_dict=SMILES_dict, gensim_model_dir=gensim_model_dir)

        database = pd.read_csv(Config_db.background_db)
        FP_table = np.load(Config_db.fp_table_path, allow_pickle=True)
        remain_features = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
        self.generator = Generator(db=db,
                                   checkpoint_path=Config_db.checkpoint_path, FP_table=FP_table, k=128,
                                   rerank_dir=Config_db.rerank_dir, database=database,
                                   remain_features=remain_features, config_path=config_path,
                                   config_generation_path=config_generation_path)


    def generate(self, spec_path, DB_cutoff=0.4, formula_path=None, mode="Free", template_path=None, rerank=True):
        Generation_dict = {}
        spec_list = self.generator.load_spec(spec_path)
        for step, spec in enumerate(spec_list):
            spec_id = get_meta(spec.metaData, "name")
            if spec_id is None:
                spec_id = step
            formula = get_meta(spec.metaData, "formula")
            if formula is None:
                raise ValueError("No formula provided in spectra data.")

            # search templates
            search_vector = self.generator.embedding(spec)
            faiss.normalize_L2(search_vector)
            templates_list = self.generator.Find_templates(formula, search_vector, DB_cutoff=DB_cutoff)

            # Generation results
            searched_res, types, modified_score, templates_list, candidates = self.generator.Generate(
                templates_list=templates_list,
                formula=formula,
                spec_id=spec_id,
                DB_cutoff=DB_cutoff,
                mode=mode, rerank=rerank)
            Generation_dict[spec_id] = (spec_id, searched_res, types, modified_score, templates_list, candidates)
        return Generation_dict

if __name__ == '__main__':
    metgenx = MetGenX(polarity="positive")
    generation_dict = metgenx.generate(
        spec_path="./test/M678T439.mgf",
        DB_cutoff=0.4,
        mode="Free"
    )