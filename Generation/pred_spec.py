
import os.path

import numpy as np
import pandas as pd
from MS2Tools.SpectrumFileReader import SpectrumFileReader
from Model.Prediction import Create_predictor
from Rerank.features import Feature_calculater
import lightgbm as lgb
from Model.datasets.smiles import SMILESStandarder
from rdkit import Chem
from collections import defaultdict


def FindCandidates(database, formula):
    candidates = list(database[database["formula"] == formula]["smiles"])
    standarder = SMILESStandarder()
    candidates = [standarder.Standard(smi, isomericSmiles=False, canonical=True, kekuleSmiles=True) for smi in
                  candidates]
    return candidates


class Generator(object):
    def __init__(self, db,
                 checkpoint_path=None, FP_table=None, k=128,
                 rerank_dir=None,
                 database=None, remain_features=None,
                 config_path="./weights/generation/config.json",
                 config_generation_path="./weights/generation/config_generation.json"
                 ):
        self.db = db
        self.ReRanker = Feature_calculater(Feature_cache=None, Cal_extra_FP=True, CanonicalTautomer=True,
                                           FP_type="Morgan")
        self.ReRankModel = lgb.Booster(model_file=rerank_dir)
        self.predictor = Create_predictor(model_dir=checkpoint_path, FP_table=FP_table, k=k, config_path=config_path,
                                          config_generation_path=config_generation_path)
        self.database = database
        self.remain_features = remain_features

    def load_spec(self, spec_path):
        spec_set = SpectrumFileReader(spec_path).read_file()
        spec_list = []
        for spec in spec_set:
            self.clean_spec(spec)
            spec_list.append(spec)
        return spec_list

    def clean_spec(self, spec):
        spec.normalize(
            normalize_to=1,
            consider_precursor=True,
            precursor_mz=spec.metaData["mz"],
            mz_tol=10,
            res_define_at=0.0
        )
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
        return spec

    def embedding(self, spec):
        search_vector = self.db.Embed_MS2(spec)
        return search_vector

    def Find_templates(self, formula, search_vector, DB_cutoff=0.4, use_modification=False, exclude_smiles=None):
        templates_list = self.db.find_templates(formula, search_vector, DB_cutoff=DB_cutoff,
                                                use_modification=use_modification, exclude_smiles=exclude_smiles)
        return templates_list

    def Predict(self, spec_id, templates_list, formula):
        searched_res = \
        self.predictor.pred_template(ID_list=[spec_id], templates=[templates_list], formula_list=[formula],
                                     Clean_formula=True)[0]
        return searched_res

    def Scoring(self, spec_id, templates_list, formula, candidates):
        searched_res = self.predictor.score_template([spec_id], [templates_list], [formula], candidates)
        return searched_res

    def Rerank(self, searched_res, templates_list, remain_features=None):
        Features = self.ReRanker.Combine_features(searched_res, templates_list, len(templates_list), len(searched_res),
                                                  remain_features=remain_features)
        modified_score = self.ReRankModel.predict(pd.DataFrame(Features))
        return modified_score

    def Generate(self, spec=None, templates_list=None, formula=None, spec_id=None, DB_cutoff=0.4, mode="Combine",
                 rerank=True):
        if spec is None and templates_list is None:
            raise ValueError("Both spectra and templates are None.")
        if spec is not None:
            search_vector = self.embedding(spec)
            templates_list = self.Find_templates(formula, search_vector, DB_cutoff=DB_cutoff, use_modification=False)

        if len(templates_list) == 0:
            return [], [], [], templates_list, []

        if mode == "Free":
            search_res = self.Predict(spec_id, templates_list, formula)
            types = ["database_free"] * len(search_res)
            candidates = []
        elif mode == "Restricted":
            if self.database is not None:
                candidates = FindCandidates(self.database, formula)
                if len(candidates) == 0:
                    return [], [], [], templates_list, candidates

                search_res = self.Scoring(spec_id, templates_list, formula, candidates)
            else:
                raise ValueError("Database is None.")
            types = ["database_restricted"] * len(search_res)

        elif mode == "Combine":
            search_res_gen = self.Predict(spec_id, templates_list, formula)
            if self.database is not None:
                candidates = FindCandidates(self.database, formula)
                if len(candidates) != 0:
                    search_res_score = self.Scoring(spec_id, templates_list, formula, candidates)
                else:
                    search_res_score = []
            else:
                raise ValueError("Database is None.")
            search_res = list(search_res_score) + list(search_res_gen)
            types = ["database_restricted"] * len(search_res_score) + ["database_free"] * len(search_res_gen)
            if len(search_res) != 0:
                sorted_indexes = np.argsort([res[1] for res in search_res])[::-1]
                search_res = [search_res[i] for i in sorted_indexes]
                types = [types[i] for i in sorted_indexes]
                grouped_data = defaultdict(list)
                for i in range(len(search_res)):
                    smi = search_res[i][0]
                    mol = Chem.MolFromSmiles(smi)
                    inchi = Chem.inchi.MolToInchiKey(mol)[0:14]
                    grouped_data[inchi].append((search_res[i], types[i]))

                types_filter = []
                search_res_filter = []
                for res in grouped_data.values():
                    gen_res = [item[0] for item in res]
                    type_res = [item[1] for item in res]
                    search_res_filter.append(gen_res[0])
                    if all(item in type_res for item in ['database_free', 'database_restricted']):
                        types_filter.append("Both")
                    else:
                        types_filter.append(type_res[0])
                search_res = search_res_filter
                types = types_filter
        else:
            raise ValueError(f"Invalid prediction type: {mode}")
        if len(search_res) == 0:
            return [], [], [], templates_list, candidates
        if rerank:
            remain_features = self.remain_features
            template_smiles = [template[0] for template in templates_list]
            template_scores = [template[1] for template in templates_list]
            template_used = list(zip(template_smiles, template_scores))
            modified_score = self.Rerank(search_res, template_used, remain_features=remain_features)
            sorted_indexes = np.argsort(modified_score)[::-1]
            search_res = [search_res[i] for i in sorted_indexes]
            types = [types[i] for i in sorted_indexes]
            modified_score = [modified_score[i] for i in sorted_indexes]
        else:
            modified_score = [None] * len(search_res)

        return search_res, types, modified_score, templates_list, candidates

    def parse_spec(self, spec_path, formula_list, ID_list=None, DB_cutoff=0.4, mode="Combine", rerank=True):
        generation_dict = {}
        spec_list = self.load_spec(spec_path)
        for idx, spec in enumerate(spec_list):
            if ID_list is None:
                spec_id = spec.metaData["name"]
            else:
                spec_id = ID_list[idx]
            search_res, types, modified_score = self.Generate(spec, formula_list[idx],
                                                              spec_id=spec_id, DB_cutoff=DB_cutoff,
                                                              mode=mode, rerank=rerank)
            generation_dict[spec_id] = (search_res, types, modified_score)
        return generation_dict
