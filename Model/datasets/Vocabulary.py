# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch
class Vocabulary(object):
    def __init__(self,
                 *,
                 pad="<pad>",
                 unk="<unk>",
                 sep="<sep>",
                 mask="<mask>",
                 special_tokens=None,
                 add_special_tokens=True):
        self.token_pad, self.token_unk, self.token_sep, self.token_mask = pad, unk, sep, mask
        self.token = []
        self.vocab = {}
        if add_special_tokens:
            self.pad_idx = self.add_token(pad)
            self.unk_idx = self.add_token(unk)
            self.sep_idx = self.add_token(sep)
            self.mask_idx = self.add_token(mask)
            if special_tokens:
                for token in special_tokens:
                    self.add_token(token)
        else:
            self.pad_idx = None

    def __getitem__(self, index):
        if index < len(self.token):
            return self.token[index]
        else:
            return self.token_unk

    def __len__(self):
        return len(self.token)

    def get_index(self, token):
        if token in self.token:
            return self.vocab[token]
        else:
            return self.unk_idx

    def add_token(self, token):
        """Adds a word to the dictionary"""
        if token in self.token:
            idx = self.vocab[token]
            return idx
        else:
            idx = len(self.token)
            self.vocab[token] = idx
            self.token.append(token)
            return idx

    def Update_vocab(self, token_list):
        self.vocab = self.vocab

        # Check_token = {"<Pad>", "<Unk>"}
        # for token in Check_token:
        #     if token not in self.vocab.keys():
        #         raise KeyError("Token ${token} is not in vocab")
        for tokens in token_list:
            for token in tokens:
                if token not in self.vocab:
                    self.add_token(token)

    def Load_vocab(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            k, v = line.strip().split(':')
            self.add_token(k)

    def Export_vocab(self, export_path):
        if self.vocab is None:
            raise KeyError("Vocab is None")

        filename = open(export_path, "w")
        for k, v in self.vocab.items():
            filename.write(k+':'+str(v)+'\n')
        filename.close()


import re
# Formula_remain = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
# vocab_size = 120
def BuildFormulaConverter(vocab,vocab_size, Formula_remain):
    rel_matrix = torch.zeros(vocab_size,len(Formula_remain))
    Vocab_keys = list(vocab.vocab.keys())
    Hydrate_num = [int(s.split('@')[1]) if '@' in s else 0 for s in Vocab_keys]
    Token_raw = [re.sub(r'@.*', '', token) for token in Vocab_keys]
    Hydrate_num.extend([0] * max(0, vocab_size - len(Hydrate_num)))
    for i in range(len(Formula_remain)):
        elements = Formula_remain[i]
        if elements != "H":
            if len(elements)==1:
                # pattern = re.compile(r'([A-Z])(?![a-z])')
                pattern = re.compile(r'(?<![A-Z])[A-Za-z](?![a-z])')
                indices = [index for index, key in enumerate(Token_raw) if (elements in key or elements.lower() in key) and len(pattern.findall(key))!=0 and "<" not in key]
            else:
                indices = [index for index, key in enumerate(Token_raw) if elements in key and "<" not in key]
            rel_matrix[indices,i] = 1
        else:
            rel_matrix[:,i] = torch.tensor(Hydrate_num)
    indices_special = [index for index, key in enumerate(vocab.vocab.keys()) if "<" in key]
    indices_special = indices_special + list(range(len(vocab)+1,vocab_size))
    indices_special = torch.tensor(indices_special)
    return rel_matrix, indices_special
