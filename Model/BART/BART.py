# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Description：
    BART model from Hugging Face
    The Bart model was proposed in BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
    Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed,
    Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.
"""
from transformers import BartConfig
import copy
import torch.nn as nn
from Model.BART.BART_transformer import BARTdecoder, Encoder
from Model.BART.base import BaseModel
from Model.BART.Generation import BeamSearch
from Model.Configs import config
class BARTModel(BaseModel):
    def __init__(self, model_config=config.Config(), use_pretrained=False, use_formula=True):
        super(BARTModel, self).__init__()
        self.save_hyperparameters()
        model_config = copy.deepcopy(model_config)
        self.config = model_config
        self.pad_idx = model_config.pad_token_id
        config_dict = model_config.to_dict()
        configuration = BartConfig(**config_dict)
        padding_idx, vocab_size = configuration.pad_token_id, configuration.vocab_size
        self.d_model = configuration.d_model

        # BART Model
        self.use_formula = use_formula
        self.encoder = Encoder(configuration, use_formula)
        self.decoder = BARTdecoder(configuration)

        # final layer
        self.lm_head = nn.Linear(configuration.d_model, vocab_size, bias=False)

        # Training_config
        if use_pretrained:
            self.load_pretrained()
        self.num_warmup = model_config.num_warmup
        self.max_steps = model_config.max_steps
        self.lr = model_config.lr
        self.weight_decay = model_config.weight_decay

    def load_pretrained(self):
        self.init_weights()
        self.pretrained_weights = self.config.pretrained_weights
        if self.pretrained_weights is None:
            raise ValueError("No pretrained weights provided")
        self.load_weights(self.pretrained_weights)

    def forward(self, batch_input: dict):
        Src_SMILES = batch_input['Src_SMILES']
        Decoder_SMILES = batch_input['dec_SMILES']
        memory, src_padding_mask = self.encoder(batch_input)
        outputs = self.decoder(memory, Decoder_SMILES)
        loss, lm_logits = self.Cal_Masked_loss(outputs=outputs, labels=Src_SMILES)
        return outputs, loss
    def decode(self, batch_input: dict, generate_config, Formula_vector):
        memory, src_padding_mask = self.encoder(batch_input)
        batch_size = len(batch_input["names"])  # batch_size
        # beam search
        batch_size = memory.size(0)  # batch_size
        # beam search
        Formula_vector = Formula_vector.float()

        res, score = BeamSearch(model=self, generate_config=generate_config, hidden_state=memory,
                                Formula_vector=Formula_vector, vocab=generate_config.vocab,
                                search_type=generate_config.search_type)
        output_num_return_sequences_per_batch = res.size(0)
        res = res.view(batch_size, output_num_return_sequences_per_batch, res.size(-1))
        score = [score[i:i + output_num_return_sequences_per_batch] for i in
                 range(0, len(score), output_num_return_sequences_per_batch)]
        return res, score
