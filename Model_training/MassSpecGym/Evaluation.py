"""
# File       : Evaluation_denovo.py
# Time       : 2025/9/18 13:43
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from massspecgym.data import MassSpecDataModule
import torch
from pytorch_lightning import Trainer
from Model.MassSpecGym.BART import BARTModel_GYM_denovo, BARTModel_GYM_Retrieval
from Model.Configs import config
from Model.datasets.Vocabulary import Vocabulary
from Model.datasets.Tokenizer import SMILESTokenizer
from pytorch_lightning import seed_everything
from pathlib import Path
import argparse

def main(args):
    if args.datasetname is not None:
        if args.Evaluation_mode == "retrieval":
            dataset_path= os.path.join("results", args.datasetname, f"input_dataset_retrieval.dataset")
            output_dir = os.path.join("results", args.datasetname, f"retrieval_result")
        else:
            dataset_path= os.path.join("results", args.datasetname, f"input_dataset.dataset")
            output_dir = os.path.join("results", args.datasetname, f"denovo_result")
        config_path = os.path.join("results", args.datasetname, f"./weights/generation/config.json")
        generate_config_path = os.path.join("results", args.datasetname, f"./weights/generation/config_generation.json")
        vocab_path = os.path.join("results", args.datasetname, f"./weights/generation/vocab.txt")
        convert_dict_path = os.path.join("results", args.datasetname, f"./weights/generation/Convert_dict.dict")
        checkpoint_path = os.path.join("results", args.datasetname, f"./weights/generation/Trained_Weight.pth.ckpt")

    else:
        dataset_path = args.dataset
        config_path = args.config
        vocab_path = args.vocab
        convert_dict_path = args.convert_dict
        generate_config_path = args.generate_config
        checkpoint_path = args.checkpoint
        output_dir = args.output_dir


    dataset = np.load(dataset_path, allow_pickle=True)
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=1, # use 1 for generation
        num_workers=args.num_workers
    )

    try:
        model_config = config.Config.from_json_file(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    vocab = Vocabulary(special_tokens=["<bos>", "<eos>"])
    vocab.Load_vocab(vocab_path)

    Convert_dict = np.load(convert_dict_path, allow_pickle=True)

    tokenizer = SMILESTokenizer(vocab=vocab, hydrate=True, Convert_dict=Convert_dict)

    generate_config = config.Config_generation.from_json_file(generate_config_path)
    generate_config.vocab = vocab

    seed_everything(42, workers=True)
    if args.Evaluation_mode == "denovo":
        model = BARTModel_GYM_denovo(
            model_config=model_config,
            use_pretrained=False,
            use_formula=True,
            # log_only_loss_at_stages=[Stage.TRAIN, Stage.VAL],
            # lr=model_config.lr,
            # weight_decay=model_config.num_warmup,
            generate_config=generate_config,
            clean_formula=False,
            SMITokenizer=tokenizer,
            generation_path=args.generation_path
        )
    elif args.Evaluation_mode == "retrieval":
        model = BARTModel_GYM_Retrieval(
            model_config=model_config,
            use_pretrained=False,
            use_formula=True,
            # log_only_loss_at_stages=[Stage.TRAIN, Stage.VAL],
            # lr=model_config.lr,
            # weight_decay=model_config.num_warmup,
            # generate_config=generate_config,
            # clean_formula=False,
            SMITokenizer=tokenizer
        )
    else:
        raise ValueError(f"Invalid mode: {args.Evaluation_mode}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    trainer = Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        logger=False,
        enable_checkpointing=False,
    )

    model.df_test_path = Path(output_dir)
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model testing with configurable arguments.")
    parser.add_argument("--datasetname", type=str, default=None, help="dataset name")
    parser.add_argument("--Evaluation_mode", choices=["denovo", "retrieval"], type=str, default="denovo",
                        help="Evaluation mode: 'denovo' for database-free evaluation, 'retrieval' for database-restricted evaluation."
                        )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.dataset file)")
    parser.add_argument("--config", type=str, default="./weights/generation/config.json",
                        help="Path to model config JSON")
    parser.add_argument("--vocab", type=str, default="./weights/generation/vocab.txt", help="Path to vocab file")
    parser.add_argument("--convert_dict", type=str, default="./weights/generation/Convert_dict.dict",
                        help="Path to Convert_dict file")
    parser.add_argument("--config_generation", type=str, default="./weights/generation/config_generation.json",
                        help="Path to generation config JSON")
    parser.add_argument("--generation_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save test results")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device for torch.load")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()
    main(args)


    ######## TESTING ########
    # class AttributeDict(dict):
    #     def __init__(self, *args, **kwargs):
    #         super(AttributeDict, self).__init__(*args, **kwargs)
    #         self.__dict__ = self

    # args = {
    #     "datasetname" : "MassSpecGym",
    #     "Evaluation_mode" : "denovo",
    #     "dataset" : "./results/MassSpecGym/input_dataset.dataset",
    #     "config" : "./weights/generation/config.json",
    #     "vocab" : "./weights/generation/vocab.txt",
    #     "convert_dict" : "./weights/generation/Convert_dict.dict",
    #     "config_generation" : "./weights/generation/config_generation.json",
    #     "generation_path" : None,
    #     "checkpoint" : "./results/MassSpecGym/Trained_Weight.pth.ckpt",
    #     "output_dir" : "./results/MassSpecGym",
    #     "num_workers" : 4,
    #     "device":"cuda",
    #     "gpu" : 0
    # }

    # args = {
    #     "datasetname" : "MassSpecGym",
    #     "Evaluation_mode" : "retrieval",
    #     "dataset" : "./results/MassSpecGym/input_dataset_retrieval.dataset",
    #     "config" : "./weights/generation/config.json",
    #     "vocab" : "./weights/generation/vocab.txt",
    #     "convert_dict" : "./weights/generation/Convert_dict.dict",
    #     "config_generation" : "./weights/generation/config_generation.json",
    #     "generation_path" : None,
    #     "checkpoint" : "./results/MassSpecGym/Trained_Weight.pth.ckpt",
    #     "output_dir" : "./results/MassSpecGym",
    #     "num_workers" : 4,
    #     "device":"cuda",
    #     "gpu" : 0
    # }
    # args = AttributeDict(args)