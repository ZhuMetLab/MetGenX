"""
# File       : 02_model_training.py
# Time       : 2025/10/23 10:25
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import numpy as np
from Model_training.data_module import MassSpecDataModule
import torch
from pytorch_lightning import Trainer
from Model.BART.BART import BARTModel
from Model.Configs import config
from Model.datasets.Vocabulary import Vocabulary
from Model.datasets.Tokenizer import SMILESTokenizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
import argparse

# adjust the number of gpus
devices = [0]

def main(args):

    if args.path_train is None:
        path_train = os.path.join("results", args.datasetname, f"input_dataset.dataset")
    else:
        path_train = args.path_train
    dataset = np.load(path_train, allow_pickle=True)

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Init model
    if args.generate_config_path is None:
        generate_config_path = os.path.join("./weights/generation/config_generation.json")
    else:
        generate_config_path = args.generate_config_path

    if args.config_path is None:
        config_path = os.path.join("./weights/generation/config.json")
    else:
        config_path = args.config_path

    print("Loading config ...")
    try:
        model_config = config.Config.from_json_file(config_path)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found.")

    vocab = Vocabulary(special_tokens=["<bos>", "<eos>"])
    vocab_path = "./weights/generation/vocab.txt"
    vocab.Load_vocab(vocab_path)

    Convert_path = "./weights/generation/Convert_dict.dict"
    Convert_dict = np.load(Convert_path, allow_pickle=True)
    tokenizer = SMILESTokenizer(vocab=vocab, hydrate=True, Convert_dict=Convert_dict)
    generate_config = config.Config_generation.from_json_file(generate_config_path)
    generate_config.vocab = vocab
    model_config.lr = args.lr
    model_config.num_warmup = args.num_warmup
    model = BARTModel(model_config=model_config, use_pretrained=False, use_formula=True)
    print("Config loaded.")
    print("Building model ...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # load pretrain model
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

    seed_everything(42, workers=True)
    if not os.path.exists(os.path.join("results",args.datasetname, "weights", "generation")):
        os.makedirs(os.path.join("results",args.datasetname, "weights", "generation"))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join("results",args.datasetname, "weights", "generation"),
        filename='Trained_Weight',
        save_weights_only=True,
        every_n_epochs=1,
        save_top_k=1,
        mode='min'
    )

    ckpt = torch.load(os.path.join("results",args.datasetname, "weights", "generation","Trained_Weight.ckpt"), map_location=device)
    torch.save(ckpt['state_dict'], os.path.join("results",args.datasetname, "weights", "generation","Trained_Weight.pth"))

    callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=5), checkpoint_callback]
    trainer = Trainer(max_steps=model_config.max_steps, accelerator=args.accelerator,
                         strategy="ddp_find_unused_parameters_true", gradient_clip_val=0.5,
                         check_val_every_n_epoch=1, devices=devices, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MetGenX model for metabolite generation")
    parser.add_argument("--datasetname", type=str, default="MassSpecGym", help="Name of dataset for training")
    parser.add_argument("--path_train", type=str, default=None, help="Dataset")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config JSON")
    parser.add_argument("--generate_config_path", type=str, default=None, help="Path to generation config JSON")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default= 5e-6, help="Learning rate")
    parser.add_argument("--num_warmup", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator: 'gpu' or 'cpu'")

    args = parser.parse_args()
    main(args)

