from transformers import BartForConditionalGeneration
import torch
from Model.BART.base import BaseModel
import math
import torch.nn as nn
class BartEmbeddingLayer(nn.Module):
    def __init__(self, config, embed_tokens: nn.Embedding = None):
        super(BartEmbeddingLayer, self).__init__()
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, config.pad_token_id)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

    def forward(self, input_ids):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        return inputs_embeds


class Encoder(BaseModel):
    def __init__(self, configuration, use_formula=True):
        super().__init__()

        self.use_formula = use_formula
        self.d_model = configuration.d_model
        self.pooling = configuration.pooling
        dense_layer = [2048, 1024, 512]
        if use_formula:
            self.model_length = configuration.n_elements+configuration.n_elements + configuration.n_fingerprint
        else:
            self.model_length = configuration.n_fingerprint

        self.dense = nn.ModuleList()
        for i in range(len(dense_layer)):
            if i == 0:
                self.dense.append(nn.Linear(self.model_length, dense_layer[i]))
            else:
                self.dense.append(nn.Linear(dense_layer[i-1], dense_layer[i]))

    def RowFusion(self,x):
        for dense in self.dense:
            x = torch.relu(dense(x))
        return x

    def _pooling_out(self, output, src_padding_mask, score, B):
        zero_mask = src_padding_mask[:, :, None].repeat(1, 1, self.d_model)
        output[zero_mask] = 0
        score = score.reshape(B, -1)
        if self.pooling == "score":
            score_sum = score.sum(1).reshape(-1, 1) + 1e-9
            score_tensor = score / score_sum
            pool_factor = score_tensor * ~src_padding_mask
        elif self.pooling == "mean":
            pool_factor = torch.clone(score).fill_(1)
            pool_factor = pool_factor * ~src_padding_mask
            # Replace all zeros with 1
            # pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        else:
            raise NotImplementedError()
        output = torch.einsum("bmn,bm->bn", output, pool_factor)
        return output

    def forward(self,batch_input:dict):
        FP_vector = batch_input['FP_vec'].float()
        num_peaks = batch_input["num_peaks"].squeeze(1)
        score = batch_input['score']
        if self.use_formula:
            Formula_vector = batch_input["form_vec"]
            Diff_vector = batch_input["diff_form_vec"]
            inputs_embeds = torch.cat([FP_vector, Formula_vector,Diff_vector], dim=2)
        else:
            inputs_embeds = FP_vector
        B, N, M = inputs_embeds.size()
        output = torch.zeros(B, N, self.d_model).to(inputs_embeds.device)
        for i in range(N):
            output[:, i, :] = self.RowFusion(inputs_embeds[:, i, :])

        # pooling
        inputs_dim = output.shape[1]
        inputs_aranged = torch.arange(inputs_dim).to(output.device)
        src_padding_mask = ~(inputs_aranged[None, :] < num_peaks[:, None])
        if self.pooling is not None:
            memory = self._pooling_out(output,src_padding_mask,score,B)
            src_padding_mask =None
        else:
            memory = output
        return memory, src_padding_mask



class BARTdecoder(BaseModel):
    def __init__(self, configuration):
        super().__init__()
        self.config = configuration
        self.pad_idx = configuration.pad_token_id
        padding_idx, vocab_size = configuration.pad_token_id, configuration.vocab_size
        self.d_model = configuration.d_model
        # BART Model
        self.decoder = BartForConditionalGeneration(configuration).get_decoder()

        # SMILES Embedding
        self.shared = nn.Embedding(vocab_size, configuration.d_model, padding_idx)
        self.decoder_embedding = BartEmbeddingLayer(configuration, self.shared)

    def forward(self, memory, Decoder_SMILES,src_padding_mask=None):
        decoder_padding_mask = torch.tensor(Decoder_SMILES != self.pad_idx)
        decoder_input_embeds = self.decoder_embedding(Decoder_SMILES)
        outputs = self.decoder(
            attention_mask=decoder_padding_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=src_padding_mask,
            inputs_embeds=decoder_input_embeds
        )
        return outputs

    # def decode(self, memory, Formula_vector, generate_config):
    #     batch_size = memory.size(0)  # batch_size
    #     # beam search
    #     Formula_vector = Formula_vector.float()
    #     res, score = BeamSearch(model=self, generate_config=generate_config, hidden_state=memory,
    #                             Formula_vector=Formula_vector, vocab=generate_config.vocab,
    #                             search_type=generate_config.search_type)
    #     output_num_return_sequences_per_batch = res.size(0)
    #     res = res.view(batch_size, output_num_return_sequences_per_batch, res.size(-1))
    #     score = [score[i:i + output_num_return_sequences_per_batch] for i in
    #              range(0, len(score), output_num_return_sequences_per_batch)]
    #     return res, score

