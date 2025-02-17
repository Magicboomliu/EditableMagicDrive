cd ../..

accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 8 tools/train.py \
  +exp=224x400 runner=8gpus runner.validation_before_run=true