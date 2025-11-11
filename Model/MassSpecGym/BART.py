"""
# File       : BART.py
# Time       : 2025/10/23 13:03
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

"""
To use this model, please install massspecgym package.

BARTModel_GYM_denovo： de novo molecule generation
BARTModel_GYM_Retrieval： retrieval molecule in candidate list

"""

from Model.datasets.data_utils import CleanSMILESList
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from Model.datasets.ProcessingFormula import generate_formula
from transformers import BartConfig
import copy
import torch.nn as nn
from Model.BART.Generation import BeamSearch

class BaseBARTModel_GYM(nn.Module):
    def __init__(self, model_config, use_pretrained=False, use_formula=True, generation_path=None, SMITokenizer=None):
        super(BaseBARTModel_GYM, self).__init__()
        self.config = copy.deepcopy(model_config)
        self.pad_idx = self.config.pad_token_id
        self.use_formula = use_formula
        self.SMITokenizer = SMITokenizer
        configuration = BartConfig(**self.config.to_dict())
        vocab_size = configuration.vocab_size
        self.d_model = configuration.d_model
        from Model.BART.BART_transformer import BARTdecoder, Encoder
        self.encoder = Encoder(configuration, use_formula)
        self.decoder = BARTdecoder(configuration)
        self.lm_head = nn.Linear(configuration.d_model, vocab_size, bias=False)

        self.num_warmup = self.config.num_warmup
        self.max_steps = self.config.max_steps
        self.lr = self.config.lr
        self.weight_decay = self.config.weight_decay

        if use_pretrained:
            self.load_pretrained()

        if generation_path is not None:
            Generate_dict = np.load(generation_path, allow_pickle=True)
            self.Generate_dict = dict(zip(Generate_dict["identifier"], Generate_dict["mols_pred"]))
        else:
            self.Generate_dict = None

    def load_pretrained(self):
        self.init_weights()
        pretrained_weights = self.config.pretrained_weights
        if pretrained_weights is None:
            raise ValueError("No pretrained weights provided")
        self.load_weights(pretrained_weights)

    def forward(self, batch_input: dict):
        Src_SMILES = batch_input['Src_SMILES']
        Decoder_SMILES = batch_input['dec_SMILES']
        memory, _ = self.encoder(batch_input)
        outputs = self.decoder(memory, Decoder_SMILES.long())
        return outputs, Src_SMILES

    def Cal_Masked_loss(self, outputs, labels):
        lm_logits = self.lm_head(outputs[0])
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=self.pad_idx)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            masked_lm_loss = None
        return masked_lm_loss, lm_logits

class BARTModel_GYM_denovo(BaseBARTModel_GYM, DeNovoMassSpecGymModel):
    def __init__(self, model_config, generate_config, clean_formula=False, *args, **kwargs):
        super().__init__(model_config, **kwargs)
        self.generation_config = copy.deepcopy(generate_config)
        self.clean_formula = clean_formula

    def decode(self, batch_input: dict, generate_config, Formula_vector):
        memory, _ = self.encoder(batch_input)
        res, score = BeamSearch(
            model=self,
            generate_config=generate_config,
            hidden_state=memory,
            Formula_vector=Formula_vector,
            vocab=generate_config.vocab,
            search_type=generate_config.search_type,
        )
        batch_size = memory.size(0)
        res = res.view(batch_size, -1, res.size(-1))
        score = [score[i:i + res.size(1)] for i in range(0, len(score), res.size(1))]
        return res, score

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        if self.Generate_dict is not None:
            loss, lm_logits = 0, 0
        else:
            outputs, Src_SMILES = self.forward(batch)
            loss, lm_logits = self.Cal_Masked_loss(outputs, Src_SMILES.long())

        if stage in self.log_only_loss_at_stages:
            mols_pred = None
        else:
            if self.Generate_dict is not None:
                ID_list = batch['identifier']
                mols_pred = [self.Generate_dict[ID] for ID in ID_list]
            else:
                formula_list = batch["formula"]
                Formula_vec = torch.stack([generate_formula(f) for f in formula_list]).squeeze(0).to(self.device)
                with torch.no_grad():
                    res, score = self.decode(batch, self.generation_config, Formula_vec)
                smiles_list = []
                for i in range(res.size(0)):
                    ids = res[i].cpu().numpy().tolist()
                    tokens = self.SMITokenizer.ids_to_tokens(ids)
                    smiles = [smi[0] for smi in self.SMITokenizer.tokens_to_string(tokens)]
                    smiles_list.append(smiles)
                mols_pred = smiles_list

            if self.clean_formula:
                cleaned_res = []
                mols_pred_clean = []
                for i, mols in enumerate(mols_pred):
                    res, _, _ = CleanSMILESList(
                        list(zip(mols, [1]*len(mols))),
                        tgt_formula=batch["formula"][i],
                        clean_formula=True,
                        clean_duplicated=True
                    )
                    cleaned_res.append(res)
                    mols_pred_clean.append([s[0] for s in res] if len(res) else mols)
                mols_pred = mols_pred_clean

        return dict(loss=loss, mols_pred=mols_pred)

class BARTModel_GYM_Retrieval(BaseBARTModel_GYM, RetrievalMassSpecGymModel):
    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        outputs, Src_SMILES = self.forward(batch)
        loss, _ = self.Cal_Masked_loss(outputs, Src_SMILES.long())

        if stage in self.log_only_loss_at_stages:
            return dict(loss=loss, scores=None)

        candidates = batch['candidates_smiles']
        score_all = []
        token_list = self.SMITokenizer.tokenize_smiles(candidates)
        token_list_dec = [["<bos>"] + token[:-1] for token in token_list]
        id_list = self.SMITokenizer.tokens_to_ids(token_list_dec)

        for smiles in id_list:
            batch["dec_SMILES"] = torch.tensor(smiles).to(self.device)
            batch["Src_SMILES"] = None
            with torch.no_grad():
                res = self(batch)[0]
                logits = self.lm_head(res[0])
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
            id_vocab = smiles[1:] + [5]
            scores = [logits[:, i, id_vocab[i]].item() for i in range(len(id_vocab))]
            score_all.append(np.mean(scores))

        return dict(loss=loss, scores=torch.tensor(score_all).to("cuda"))

