cd ..

accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 1 tools/train.py \
  +exp=224x400_lidar runner=debug 
  
  
  # runner.validation_before_run=true