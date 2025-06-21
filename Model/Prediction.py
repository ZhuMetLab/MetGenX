import os
import torch
import numpy as np
from Model.datasets import data_utils
from Model.datasets.ProcessingFormula import generate_formula
from Model.datasets.Tokenizer import SMILESTokenizer
from Model.datasets.data_utils import CleanSMILESList, get_free_gpu
from Model.Fingerprints.fingerprinting import Fingerprinter,get_fp
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from Model.Configs import config
from Model.BART.BART import BARTModel

class Predictor(object):
    def __init__(self, generate_config, model, FP_table, SMITokenizer,Cal_extra_FP=False, used_index=None):
        self.model = model
        self.generate_config = generate_config
        self.FP_dict =dict(zip(FP_table["CpdID"], FP_table["fingerprint"]))
        self.formula_dict = dict(zip(FP_table["CpdID"], FP_table["Formula"]))
        self.Convert_dict = dict(zip(FP_table["InChI1"], FP_table["CpdID"]))
        self.SMITokenizer = SMITokenizer
        self.Cal_extra_FP = Cal_extra_FP
        self.FP_table = FP_table
        if Cal_extra_FP:
            self.FP_cache = []
            self.Fingerprinter = Fingerprinter(lib_path="./Model/Fingerprints/fingerprint-wrapper/target/fingerprint-wrapper-bin-0.5.2.jar")
            self.used_index = used_index

    def pred_template(self, ID_list, templates, formula_list, Clean_formula=True):
        batch_input = [self._construct_input(templates[i],formula_list[i]) for i in range(len(templates))]
        for k in range(len(batch_input)):
            batch_input[k]["names"] = ID_list[k]
        searched_res, _, _ = self.pred(batch_input, formula_list, Clean_formula=Clean_formula)
        return searched_res

    def score_template(self, ID_list, templates, formula_list, candidates):
        batch_input = [self._construct_input(templates[i], formula_list[i]) for i in range(len(templates))]
        for k in range(len(batch_input)):
            batch_input[k]["names"] = ID_list[k]
        searched_res = self.scoring(batch_input, candidates)
        return searched_res

    def scoring(self, batch_input, candidates):
        batch_input = self.collater(batch_input)
        for key, value in batch_input.items():
            if value is not None and isinstance(value, torch.Tensor):
                batch_input[key] = value.to(self.model.device)
        score_all = []
        token_list = self.SMITokenizer.tokenize_smiles(candidates)
        token_list_dec = [["<bos>"] + token[:-1] for token in token_list]
        id_list = self.SMITokenizer.tokens_to_ids(token_list_dec)
        for smiles in id_list:
            batch_input["dec_SMILES"] = torch.tensor(smiles).to(self.model.device)
            batch_input["Src_SMILES"] = None
            with torch.no_grad():
                res = self.model(batch_input)[0]
                logits = self.model.lm_head(res[0])
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
            id_vocab = np.array(smiles).tolist()
            id_vocab = id_vocab[1:] + [5]
            scores = []
            for i in range(len(id_vocab)):
                scores.append(logits[:, i, id_vocab[i]].item())
            score = np.mean(scores)
            score_all.append(score)
        searched_res = sorted(zip(candidates, score_all), key=lambda x: x[1], reverse=True)
        return searched_res

    def pred(self, batch_input, formula_list, Clean_formula=True):
        batch_input = self.collater(batch_input)
        Formula_vec = torch.stack([generate_formula(formula) for formula in formula_list]).squeeze(0)
        Formula_vec = Formula_vec.to(self.model.device)
        for key, value in batch_input.items():
            if value is not None and isinstance(value, torch.Tensor):
                batch_input[key] = value.to(self.model.device)
        with torch.no_grad():
            res, score = self.model.decode(batch_input, self.generate_config, Formula_vec)
        res = res.cpu()
        batch_size = res.size(0)
        smiles_list = []
        for batch_idx in range(batch_size):
            searched_ids = res[batch_idx, :, :]
            id_list = searched_ids.numpy().tolist()
            token_list = self.SMITokenizer.ids_to_tokens(id_list)
            smiles = self.SMITokenizer.tokens_to_string(token_list)
            smiles = [smi[0] for smi in smiles]
            smiles_list.append(smiles)
        searched_res = [list(zip(smiles_list[i], score[i])) for i in range(len(smiles_list))]
        if Clean_formula:
            num_vaild_smi = []
            num_correct_formula = []
            cleaned_res = []
            for i in range(len(searched_res)):
                res, num_vaild_smi_i, num_correct_formula_i = CleanSMILESList(searched_res[i], tgt_formula=formula_list[i],
                                            clean_formula=True, clean_duplicated=True)
                cleaned_res.append(res)
                num_vaild_smi.append(num_vaild_smi_i)
                num_correct_formula.append(num_correct_formula_i)
            searched_res = cleaned_res
        else:
            num_vaild_smi = "NA"
            num_correct_formula = "NA"

        searched_res = list(searched_res)
        return searched_res, num_vaild_smi, num_correct_formula

    def collater(self, samples,pad_idx=0, pad_fixed_length=None):
        if len(samples) == 0:
            return {}
        batch = {}
        for k, v in samples[0].items():
            if isinstance(v, torch.Tensor):
                batch[k] = data_utils.collater(
                    [sample[k] for sample in samples], pad_idx, pad_fixed_length)
            else:
                batch[k] = [sample[k] for sample in samples]
        return batch

    def _Convert_Cpd_id(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        InChI1 = Chem.inchi.MolToInchiKey(mol)[0:14]
        if InChI1 in self.Convert_dict.keys():
            Cpdid = self.Convert_dict[InChI1]
        else:
            if self.Cal_extra_FP:
                Cpdid = "New"+str(len(self.FP_cache))
                self.FP_cache.append(Cpdid)
                self.Convert_dict[InChI1] = Cpdid
            else:
                raise ValueError("Unknown smiles included. Please open Cal_extra_FP.")
        return Cpdid, InChI1

    def _retrive_FP(self, smiles):
        cpd_id, InChI1 = self._Convert_Cpd_id(smiles)
        if cpd_id in self.FP_dict.keys():
            FP = self.FP_dict[cpd_id]
            formula = self.formula_dict[cpd_id]
        else:
            formula = CalcMolFormula(Chem.MolFromSmiles(smiles))
            FP = self.Fingerprinter.process([smiles])
            FP = get_fp(FP[0]["fingerprint"])[0][self.used_index]
            self.FP_dict[cpd_id] = FP
            self.formula_dict[cpd_id]=formula
            self.FP_table.loc[len(self.FP_table.index)] = [cpd_id, formula, FP, InChI1]
        return FP, formula

    def _construct_input(self, template_list, tgt_formula):
        num_peaks = len(template_list)
        form_raw = generate_formula(tgt_formula)
        score = [Template[1] for Template in template_list]
        smi_id = [Template[0] for Template in template_list]
        Pairs = [self._retrive_FP(smiles) for smiles in smi_id]
        FP_vec = torch.tensor(np.array([pair[0] for pair in Pairs]))
        form_vec = torch.stack([generate_formula(pair[1]) for pair in Pairs]).squeeze(1)

        diff_form_vec = form_vec - form_raw
        batch_dict = {
            "diff_form_vec": diff_form_vec,
            "form_vec": form_vec,
            "score": torch.tensor(score),
            "template_names": [smi_id],
            "num_peaks": torch.tensor([num_peaks]),
            "FP_vec": FP_vec
        }
        return batch_dict


def Create_predictor(model_dir, FP_table, k=None,
                     config_path="./weights/generation/config.json",
                     config_generation_path="./weights/generation/config_generation.json"):
    from Model.datasets.Vocabulary import Vocabulary
    vocab = Vocabulary(special_tokens=["<bos>", "<eos>"])
    vocab_path = "./weights/generation/vocab.txt"
    vocab.Load_vocab(vocab_path)
    print("Vocabulary loaded, size: {}".format(len(vocab)))
    model_config = config.Config.from_json_file(config_path)
    generate_config = config.Config_generation.from_json_file(config_generation_path)
    generate_config.convert_from_model(model_config)
    if k is not None:
        generate_config.num_beams = k
    # generate_config.length_penalty = 1
    # generate_config.max_length = 128
    # generate_config.search_type = 'ConstraintBeam_search'
    # generate_config.search_type = 'Beam_search'
    generate_config.vocab = vocab
    model = BARTModel(model_config=model_config, use_pretrained=False)
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        device = torch.device(get_free_gpu())
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    model.to(device)
    vocab = generate_config.vocab
    Convert_dict = np.load("./weights/generation/Convert_dict.dict", allow_pickle=True)
    SMITokenizer = SMILESTokenizer(vocab=vocab, hydrate=True, Convert_dict=Convert_dict)
    checkpoint_path = model_dir
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(model.device))
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    import pandas as pd
    import os
    fingerid_df = pd.read_table(os.path.join("./weights/generation/Fingerprints.tsv"))
    used_index = list(fingerid_df.absoluteIndex)
    predictor = Predictor(generate_config=generate_config, model=model, FP_table=FP_table, SMITokenizer=SMITokenizer,
                          Cal_extra_FP=True, used_index=used_index)
    return predictor


