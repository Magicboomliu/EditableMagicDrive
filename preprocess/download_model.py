from diffusers import StableDiffusionPipeline
import torch
 
target_folder = "/home/Zihua/DEV/MagicDrive/pretrained"  # Change this to your desired location
model_id = "botp/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=target_folder)
pipe.to("cuda")  # Use GPU if available
 
print(f"Model downloaded to: {target_folder}")