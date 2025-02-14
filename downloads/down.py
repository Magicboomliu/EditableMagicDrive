from diffusers import StableDiffusionPipeline
import torch

target_folder = "/home/SonyInternship/MagicDrive/SD15"  # Change this to your desired location
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=target_folder)
pipe.to("cuda")  # Use GPU if available

print(f"Model downloaded to: {target_folder}")