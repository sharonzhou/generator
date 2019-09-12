#!/bin/bash
python train.py --use_custom_input_noise --disable_batch_norm --image_name hexes.png --mask_name 512_square.png --gpu_ids 3 --name debug_z_test
