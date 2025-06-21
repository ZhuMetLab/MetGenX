import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import torch.nn as nn
class BaseModel(pl.LightningModule):
    def forward(self):
        raise NotImplementedError()

    def _prepare_model_input(self,
                             input_ids: torch.LongTensor = None,
                             attention_mask: torch.Tensor = None,
                             decoder_input_ids: torch.LongTensor = None,
                             decoder_attention_mask: torch.Tensor = None,
                             labels: torch.LongTensor = None,
                             inputs_embeds: torch.Tensor = None,
                             decoder_inputs_embeds: torch.Tensor = None
                             ):
        model_input = {var_name: var_value for var_name, var_value in locals().items()
                       if var_name != 'self' and var_value is not None}
        return model_input

    def Cal_Masked_loss(self, outputs, labels):
        lm_logits = self.lm_head(outputs[0])
        # lm_logits = lm_logits + self.bart_model.final_logits_bias.to(lm_logits.device)
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=self.pad_idx)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return masked_lm_loss, lm_logits

    def training_step(self, batch, batch_idx):
        # (self.current_epoch)
        output, loss = self(batch)
        # model_output = self(batch)
        # loss = model_output.loss
        # loss.requires_grad_(True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self(batch)
        # loss = model_output.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        output, loss = self(batch)
        # loss = model_output.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    # do something with these
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup,
                                                    num_training_steps=self.max_steps)
        return [optimizer], [scheduler]

    def init_weights(self):
        for name, module in self._modules.items():
            self._init_weights(module)

    def _init_weights(self, module: torch.nn.Module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def load_weights(self, dir_weights):
        model_weights = torch.load(dir_weights, map_location=torch.device(self.device))
        model_weights = model_weights["state_dict"]
        model_dict = self.state_dict()
        pretrain_weights = {k: v for k, v in model_weights.items() if k in model_dict}
        print("Total {} weights were loaded.".format(len(pretrain_weights)))
        model_dict.update(pretrain_weights)
        self.load_state_dict(model_dict)
