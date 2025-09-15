# LID-VTON
This repository contains the official implementation of the paper "LID-VTON" submitted to The Visual Computer journal.

# Requirements
The code is tested with the following environment:
Python 3.8+
PyTorch 1.10.1
torchvision 0.11.2
CUDA 11.x

# Dataset Preparation
Download the VITON-HD dataset

# Testing
The testing process consists of two stages: warping and try-on generation.
# Stage 1: Warping
bash
python3 -m torch.distributed.run test_warping.py \
    --name test_partflow_vitonhd_unpaired_1109 \
    --PBAFN_warp_checkpoint 'checkpoints/LID-VTON_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
    --resize_or_crop None --verbose --tf_log \
    --dataset vitonhd --resolution 512 \
    --batchSize 2 --num_gpus 1 --label_nc 14 --launcher pytorch \
    --dataroot /path/to/your/VITON-HD-512 \
    --image_pairs_txt test_pairs_unpaired_1018.txt
# Stage 2: Try-On Generation
bash
python3 -m torch.distributed.run test_tryon.py \
    --name test_gpvtongen_vitonhd_unpaired_1109 \
    --resize_or_crop None --verbose --tf_log \
    --dataset vitonhd --resolution 512 \
    --batchSize 6 --num_gpus 1 --label_nc 14 --launcher pytorch \
    --PBAFN_gen_checkpoint 'checkpoints/LID-VTON_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth' \
    --dataroot /path/to/your/VITON-HD-512 \
    --image_pairs_txt test_pairs_unpaired_1018.txt \
    --warproot sample/test_partflow_vitonhd_unpaired_1109
