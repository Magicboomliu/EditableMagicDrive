cd ../..

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 2 tools/train.py \
  +exp=224x400_with_lidar runner=8gpus_small_with_lidar_version1 



# runner.validation_before_run=true