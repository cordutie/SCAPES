#!/usr/bin/env bash
set -e
echo "=== Experiment 1/4: Baseline (no reg, no GAN) ==="
python 2_training.py \
    --path /home/esteban/Documents/projects/dev_SCAPES/datasets/microtex_1 \
    --save_path /home/esteban/Documents/projects/dev_SCAPES/models/SCAPES
echo "=== Experiment 2/4: Regularizers only ==="
python 2_training.py \
    --path /home/esteban/Documents/projects/dev_SCAPES/datasets/microtex_2 \
    --save_path /home/esteban/Documents/projects/dev_SCAPES/models/SCAPES_reg
echo "=== Experiment 3/4: GAN only ==="
python 2_training.py \
    --path /home/esteban/Documents/projects/dev_SCAPES/datasets/microtex_3 \
    --save_path /home/esteban/Documents/projects/dev_SCAPES/models/SCAPES_gan
echo "=== Experiment 4/4: GAN + Regularizers ==="
python 2_training.py \
    --path /home/esteban/Documents/projects/dev_SCAPES/datasets/microtex_4 \
    --save_path /home/esteban/Documents/projects/dev_SCAPES/models/SCAPES_gan_reg
echo "=== All experiments complete! ==="