#!/bin/bash
python Model_training/MassSpecGym/Evaluation.py \
    --datasetname "MassSpecGym" \
    --Evaluation_mode "denovo" \
    --dataset "./results/MassSpecGym/input_dataset.dataset" \
    --config "./weights/generation/config.json" \
    --vocab "./weights/generation/vocab.txt" \
    --convert_dict "./weights/generation/Convert_dict.dict" \
    --config_generation "./weights/generation/config_generation.json" \
    --checkpoint "./results/MassSpecGym/Trained_Weight.pth.ckpt" \
    --output_dir "./results/MassSpecGym" \
    --num_workers 4 \
    --device "cuda" \
    --gpu 0