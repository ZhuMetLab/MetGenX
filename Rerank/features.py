
import numpy as np
from rdkit.Chem import AllChem
import pandas as pd
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit import Chem
from Rerank.NP_score import npscorer
from Model.Fingerprints.fingerprinting import Fingerprinter
import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        # Close the saved file descriptors
        os.close(self.save_fds[0])
        os.close(self.save_fds[1])


class Feature_calculater(object):
    def __init__(self, Feature_cache=None,FP_cache=None, Cal_extra_FP=True, standard_smiles=True, used_index=None,
                 FP_type="Daylight", CanonicalTautomer=False):
        self.Cal_extra_FP = Cal_extra_FP
        fscore = npscorer.readNPModel(filename="./Rerank/NP_score/publicnp.model.gz")
        self.fscore = fscore
        if not Feature_cache:
            Feature_cache = {}
        self.Feature_cache = Feature_cache
        if not FP_cache:
            FP_cache = {}
        self.FP_cache = FP_cache
        if Cal_extra_FP:
            self.Fingerprinter = Fingerprinter(
                lib_path="./Model/Fingerprints/fingerprint-wrapper/target/fingerprint-wrapper-bin-0.5.2.jar")
            # self.used_index = used_index
        self.standard_smiles = standard_smiles
        if standard_smiles:
            from Model.datasets.smiles import SMILESStandarder
            self.standarder = SMILESStandarder()
        self.FP_type = FP_type
        self.CanonicalTautomer = CanonicalTautomer

    def Cal_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        InChI1 = Chem.inchi.MolToInchiKey(mol)[0:14]
        if InChI1 in self.Feature_cache.keys():
           feature = self.Feature_cache[InChI1]
        else:
            feature = {}
            # Calculate NP score
            NP_score = npscorer.scoreMol(mol,self.fscore)
            feature["NP_score"] = NP_score

            # Molecular complexity
            complexity = BertzCT(mol, cutoff=100, dMat=None, forceDMat=1)
            feature["complexity"] = complexity
        return feature

    def Cal_FP(self, smiles, FP_type="Daylight"):
        mol = Chem.MolFromSmiles(smiles)
        InChI1 = Chem.inchi.MolToInchiKey(mol)[0:14]
        if InChI1 in self.FP_cache.keys():
           FP = self.FP_cache[InChI1]
        else:
            if FP_type == "Daylight":
                fpgen = AllChem.GetRDKitFPGenerator(fpSize=2048)
                FP = fpgen.GetFingerprint(mol)
            elif FP_type == "Morgan":
                FP = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=False,
                                                            useChirality=False)
            else:
                raise ValueError(f"Invalid fingerprint type {FP_type}")

            # with suppress_stdout_stderr():
            #     FP = self.Fingerprinter.process([smiles])
            #     FP = get_fp(FP[0]["fingerprint"])[0]
        return FP

    def CalSimilarity(self, fp1, fp2, similar_type="Tanimoto"):
        score = AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
        # if similar_type=="Tanimoto":
        #     score = jaccard_score(fp1, fp2, average='micro')
        # else:
        #     raise ValueError("Invalid similar type.")
        return score

    def CalTemplateScore(self, candidate_smiles, template_smiles, template_scores):
        sim_scores = []
        fp1 = self.Cal_FP(candidate_smiles, FP_type=self.FP_type)
        for smiles in template_smiles:
            fp2 = self.Cal_FP(smiles, FP_type=self.FP_type)
            sim_scores.append(self.CalSimilarity(fp1,fp2, similar_type="Tanimoto"))
        if sum(sim_scores)==0:
            average_sim_score = 0
        else:
            average_sim_score = sum([x * y / sum(sim_scores) for x, y in zip(sim_scores, template_scores)])
        average_template_score = sum([x * y/sum(template_scores) for x, y in zip(sim_scores, template_scores)])
        max_spec_score = max(template_scores)
        min_spec_score = min(template_scores)
        mean_spec_score = np.mean(template_scores)
        max_template_score = max(sim_scores)
        min_template_score = min(sim_scores)
        mean_template_score = np.mean(sim_scores)
        weighted_score = sum([sigmoid(x * y, beta=-9, r=0.6) for x, y in zip(sim_scores, template_scores)])
        weighted_score = weighted_score/len(template_smiles)
        weighted_score_feature = [average_template_score, average_sim_score,
                                  max_spec_score,min_spec_score,mean_spec_score,
                                  max_template_score,mean_template_score, min_template_score,
                                  weighted_score]
        weighted_score_feature = [round(score,2) for score in weighted_score_feature]
        return weighted_score_feature

    def Cal_mol_features(self, candidate_smiles,model_score, template_smiles, template_scores):
        weighted_score_feature = self.CalTemplateScore(candidate_smiles, template_smiles, template_scores)
        feature = self.Cal_features(candidate_smiles)
        NP_score = round(feature["NP_score"],4)
        complexity = round(feature["complexity"],4)
        smiles_length = len(candidate_smiles)
        weighted_score_feature.extend([model_score,NP_score,complexity,smiles_length])
        return weighted_score_feature

    def Combine_features(self,candidates, template_used, n_templates, n_candidates, remain_features=None):
        """
        feature index
        average_template_score	0
        average_sim_score	1
        max_spec_score	2
        min_spec_score	3
        mean_spec_score	4
        max_template_score	5
        mean_template_score	6
        min_template_score	7
        weighted_score 8
        model_score	9
        NP_score	10
        complexity	11
        smiles_length	12
        step	13
        n_templates	14
        n_candidates	15
        """
        template_smiles, template_scores = zip(*template_used)
        if self.standarder:
            template_smiles = [self.standarder.Standard(smi, isomericSmiles=False, canonical=True, kekuleSmiles=True, CanonicalTautomer=self.CanonicalTautomer) for
                               smi in template_smiles]

        Features = []
        for step, (candidate_smiles, model_score) in enumerate(candidates):
            if self.standarder:
                candidate_smiles = self.standarder.Standard(candidate_smiles, isomericSmiles=False, canonical=True, kekuleSmiles=True, CanonicalTautomer=self.CanonicalTautomer)
            mol_features = self.Cal_mol_features(candidate_smiles,model_score,template_smiles,template_scores)
            mol_features.append(step)
            mol_features.append(n_templates)
            mol_features.append(n_candidates)
            Features.append(mol_features)
        if remain_features:
            Features = [[feature[i] for i in remain_features] for feature in Features]
        return Features

    def Cal_confidence_features(self, template_used, n_templates, n_candidates):
        template_smiles, template_scores = zip(*template_used)
        max_spec_score = max(template_scores)
        min_spec_score = min(template_scores)
        mean_spec_score = np.mean(template_scores)
        Features = [max_spec_score,min_spec_score,mean_spec_score, n_templates, n_candidates]
        return Features

    def Cal_labels(self, candidates, tgt_smi, FP_type="Daylight"):
        if self.standarder:
            tgt_smi = self.standarder.Standard(tgt_smi, isomericSmiles=False, canonical=True,
                                                        kekuleSmiles=True, CanonicalTautomer=self.CanonicalTautomer)
        labels = []
        for step, (candidate_smiles, _) in enumerate(candidates):
            fp1 = self.Cal_FP(candidate_smiles,FP_type=FP_type)
            fp2 = self.Cal_FP(tgt_smi,FP_type=FP_type)
            labels.append(self.CalSimilarity(fp1, fp2, similar_type="Tanimoto"))
        return labels

import math
def sigmoid(x, beta=1, r=0):
    x = beta*(x-r)
    return 1 / (1 + math.exp(x))


