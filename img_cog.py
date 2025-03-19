from diffusers import CogView4Pipeline
import torch
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--style')
args = parser.parse_args()

if(args.style == 'se'):
  file = 'prompts_se.txt'
else:
  file = 'prompts_g.txt'

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16).to('cuda')

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

with open(file) as f:
  for prompt in f:
    path = os.path.join('imgs_cog', args.style, prompt.replace(" ","_").replace("\n",""))
    os.makedirs(path, exist_ok=True)
    for i in range(20):
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            num_inference_steps=50,
            width=512,
            height=512,
        ).images[0]

        image.save(os.path.join(path, f"{prompt}_{i}.png"))
