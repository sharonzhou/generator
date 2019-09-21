#!/bin/bash
python train.py --use_custom_input_noise --disable_batch_norm --gpu_ids 0,1,2,3 --name debug_z_viz
