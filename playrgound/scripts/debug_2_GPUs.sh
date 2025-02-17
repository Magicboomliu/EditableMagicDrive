cd ../..

accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 2 tools/train.py \
  +exp=224x400 runner=debug runner.validation_before_run=true