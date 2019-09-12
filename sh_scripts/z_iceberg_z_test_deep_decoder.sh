#!/bin/bash
python train.py --use_custom_input_noise --image_name z_iceberg.png --mask_name 512_square.png --z_test_image_name lena.png --gpu_ids 3 --name z_iceberg_z_test_deep_decoder_net
