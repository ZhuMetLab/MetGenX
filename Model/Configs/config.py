# !/usr/bin/env python
# -*-coding:utf-8 -*-


import copy
import json
import six


class Config_Base(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        config = cls()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def update_dict(cls, dict_object):
        config = cls()
        for key, value in dict_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def output_json(self, path):
        config_json = self.to_dict()
        with open(path, 'w') as json_file:
            json.dump(config_json, json_file)


class Config(Config_Base):
    def __init__(self,
                 vocab_size=120,  #
                 max_position_embeddings=150,
                 encoder_layers=6,
                 encoder_ffn_dim=3072,
                 encoder_attention_heads=8,
                 decoder_layers=6,
                 decoder_ffn_dim=3072,
                 decoder_attention_heads=8,
                 encoder_layerdrop=0.0,
                 decoder_layerdrop=0.0,
                 activation_function="gelu",
                 n_elements=10,
                 d_model=512,
                 dropout=0.1,
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 init_std=0.02,
                 classifier_dropout=0.0,
                 scale_embedding=False,
                 use_cache=True,
                 num_labels=3,
                 pad_token_id=0,  #
                 bos_token_id=4,  #
                 eos_token_id=5,  #
                 is_encoder_decoder=True,
                 decoder_start_token_id=2,
                 forced_eos_token_id=5,
                 n_fingerprint=1024,
                 pretrained_weights=None,
                 formula_idx=6,
                 SMILES_idx=9,
                 spec_idx=7,
                 template_idx=8,
                 num_warmup=4000,
                 max_steps=300000,
                 lr=0.01,
                 weight_decay=1e-6,
                 pooling="mean",
                 **kwargs):
        super(Config).__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.n_elements = n_elements
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id  #
        self.bos_token_id = bos_token_id  #
        self.eos_token_id = eos_token_id  #
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id  #
        self.pretrained_weights = pretrained_weights
        self.n_fingerprint = n_fingerprint
        self.formula_idx = formula_idx
        self.SMILES_idx = SMILES_idx
        self.spec_idx = spec_idx
        self.template_idx = template_idx
        self.template_idx = template_idx
        self.num_warmup = num_warmup
        self.max_steps = max_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.pooling = pooling


class Config_generation(Config_Base):
    def __init__(self,
                 search_type='beam_search',
                 batch_size=8,
                 num_beams=10,
                 return_topk=None,
                 max_length=20,
                 sos_token_id=4,
                 vocab_size=120,
                 bos_token_id=4,
                 pad_token_id=0,
                 eos_token_id=5,
                 cur_len=1,
                 length_penalty=1.0,
                 vocab=None):
        super().__init__()
        self.search_type = search_type
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.return_topk = return_topk
        self.max_length = max_length
        self.sos_token_id = sos_token_id
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.cur_len = cur_len
        self.length_penalty = length_penalty
        self.vocab = vocab

    def convert_from_model(self, config):
        for k, v in vars(self).items():
            if k in vars(config).keys():
                setattr(self, k, getattr(config, k))


class Config_databse(Config_Base):
    def __init__(self,
                 MetInfo_pos="./weights/positive/database_positive.db",
                 MetInfo_neg="./weights/negative/database_negative.db",
                 spec_pos="./weights/positive/embedding_spectra_NCE30_positive.db",
                 spec_neg="./weights/negative/embedding_spectra_NCE30_negative.db",
                 embed_pos="./weights/positive/w2v_spec_positive.w",
                 embed_neg="./weights/negative/w2v_spec_negative.w",
                 rerank_dir="./weights/rerank/checkpoint_gbm",
                 background_db="./database/Background_database.csv",
                 fp_table_path="./database/FP_table.pkl",
                 checkpoint_path="./weights/generation/Trained_Weight_20250522.pth",
                 ):
        super().__init__()
        self.MetInfo_pos = MetInfo_pos
        self.MetInfo_neg = MetInfo_neg
        self.spec_pos = spec_pos
        self.spec_neg = spec_neg
        self.embed_pos = embed_pos
        self.embed_neg = embed_neg
        self.rerank_dir = rerank_dir
        self.background_db = background_db
        self.fp_table_path = fp_table_path
        self.checkpoint_path = checkpoint_path

if __name__ == '__main__':
    config = Config()
    config.output_json("./config.json")
    dict = config.to_dict()

    config2 = Config_databse("/mnt/local-disk1/data/Wanghongmiao/MetGenX/weights/generation/config_database.json")
    config.output_json("/mnt/local-disk1/data/Wanghongmiao/MetGenX/weights/generation/config_database.json")
    dict = config.to_dict()
