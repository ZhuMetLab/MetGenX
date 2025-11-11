#!/bin/bash
cd "$(dirname "$0")/.."
python ./Model_training/training.py \
  --datasetname MassSpecGym \
  --path_train ./results/MassSpecGym/input_dataset.dataset \
  --checkpoint_path ./weights/Pretrained_Weight_MetGenX.pth \
  --batch_size 64 \
  --lr 5e-6 \
  --accelerator gpu \
  --num_workers 4