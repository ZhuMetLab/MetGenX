# !/usr/bin/env python
# -*-coding:utf-8 -*-

import re
import torch
class BaseTokenizer(object):
    def __init__(self,
                 *,
                 vocab=None,
                 spe_vocab=None,
                 special_tokens=["<bos>", "<eos>", "<pad>"],
                 hydrate=False,
                 Convert_dict=None

                 ):
        super().__init__()
        self.vocab = vocab
        self.spe_vocab = spe_vocab
        self.special_tokens = special_tokens
        self.hydrate = hydrate
        self.Convert_dict = Convert_dict
    def Create_Convert_dict(self, dir):
        import pandas as pd
        from Model.datasets.Vocabulary import BuildFormulaConverter
        vocab = self.vocab
        Formula_remain = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
        rel_matrix, indices_special = BuildFormulaConverter(vocab, len(vocab), Formula_remain)
        rel_matrix = pd.DataFrame(rel_matrix)
        Convert_dict = {}
        for i in range(len(Formula_remain)):
            index_ones = rel_matrix.loc[rel_matrix[i] == 1].index
            for indice in index_ones:
                Convert_dict[vocab[indice]] = Formula_remain[i]
        import pickle
        with open(dir,"wb") as f:
            pickle.dump(Convert_dict, f)

    def tokens_to_ids(self, token_list, pad=False):
        id_list = []
        max_len = max(len(tokens) for tokens in token_list)
        for tokens in token_list:
            ids = [self.vocab.get_index(token) for token in tokens]
            if pad:
                ids.extend([self.vocab.pad_idx] * (max_len - len(ids)))
            id_list.append(ids)
        return id_list

    def ids_to_tokens(self, id_list):
        if isinstance(id_list, torch.Tensor):
            id_list = id_list.numpy().tolist()
        token_list = []
        # reverse_vocab = {v: k for k, v in self.vocab.items()}
        for ids in id_list:
            tokens = [self.vocab[id] for id in ids]
            tokens = [re.sub(r'@.*', '', token) for token in tokens]
            token_list.append(tokens)

        return token_list

    def tokens_to_string(self, token_list):
        smiles_list = []
        for tokens in token_list:
            tokens = [token for token in tokens if token not in self.special_tokens]
            smiles_list.append([''.join(tokens)])
        return smiles_list

from rdkit import Chem
class SMILESTokenizer(BaseTokenizer):
    # def __init__(self, vocab):
    #     super(SMILESTokenizer).__init__()

    def tokenize_smiles(self, smiles_list):
        Token_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|\\#|\\||@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
        token_list = []
        for smiles in smiles_list:
            token = re.findall(Token_pattern, smiles)
            if self.hydrate:
                mol = Chem.MolFromSmiles(smiles)
                hydrogen_counts = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
                # Attached hydrogen number depends on the original smiles.
                attach_hydrogen = []
                i = 0
                for atom in token:
                    if atom in list(self.Convert_dict.keys()):
                        attach_hydrogen.append(hydrogen_counts[i])
                        i = i + 1
                    else:
                        attach_hydrogen.append(0)
                token = ['@'.join(map(str, pair)) for pair in zip(token, attach_hydrogen)]
            token.append("<eos>")
            token_list.append(token)
        return token_list

    def tokenize(self, smiles_list, pad=True, return_tensors=True):
        tokens = self.tokenize_smiles(smiles_list)
        ids = self.tokens_to_ids(tokens, pad=pad)
        mask = [[1 if token != self.vocab.pad_idx else 0 for token in id_seq] for id_seq in ids]
        if return_tensors:
            ids = torch.tensor(ids)
            mask = torch.tensor(mask)
        tokens_output = {"input_ids": ids, "attention_mask": mask}
        return tokens_output