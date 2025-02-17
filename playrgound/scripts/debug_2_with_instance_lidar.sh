cd ../..

accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 8 tools/train.py \
  +exp=224x400_with_instance_lidar runner=8gpus_small_with_lidar_version1 


