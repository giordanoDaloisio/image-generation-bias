import torch
from diffusers import (
    StableDiffusion3Pipeline,
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
)
from huggingface_hub import login
from argparse import ArgumentParser
import os
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--stable_version", default="3", choices=["2", "3", "xl", "chroma"])
parser.add_argument("--num_imgs", default=20, type=int)
parser.add_argument("--data")
parser.add_argument("--type", choices=["General", "SE"])
parser.add_argument('--fair', action='store_true')

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on ", device)

if args.stable_version == "3":
    login("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")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    pipe.to(device)
    folder = os.path.join("Images", args.type, "imgs3")
    timefile = "times3"
    emissionfile = "emissions3.csv"

elif args.stable_version == "xl":
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    folder = os.path.join("Images", args.type, "imgsxl")
    timefile = "timesxl"
    emissionfile = "emissionsxl.csv"

else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2", torch_dtype=torch.float16
    ).to(device)
    folder = os.path.join("Images", args.type, "imgs2")
    timefile = "times2"
    emissionfile = "emissions2.csv"

if args.fair:
    folder = folder + "_fair"

if device == "cuda":
    pipe.enable_sequential_cpu_offload()
os.makedirs(folder, exist_ok=True)

# data = pd.read_csv(args.data, index_col=0)

prompts = open(args.data, "r")
lines = prompts.readlines()

times = pd.DataFrame(columns=["time"])
if "Software Engineer" in lines[0]:
    output_file = "Stats/SE_" + emissionfile
    timefile = "Stats/SE_" + timefile
else:
    output_file = "Stats/G_" + emissionfile
    timefile = "Stats/G_" + timefile


# Remove old files
if os.path.exists("Stats/G_" + timefile):
    os.remove("Stats/G_" + timefile)
if os.path.exists("Stats/SE_" + timefile):
    os.remove("Stats/SE_" + timefile)
# if os.path.exists(timefile):
#     os.remove(timefile)
# for img in os.listdir(folder):
#     os.remove(os.path.join(folder,img))


for prompt in lines:
    # prompt = "Photo portrait of a " + p
    # print(prompt)
    # Timing functions
    if args.fair:
        prompt = prompt.replace('\n','') + ", such that it fairly represents different genders and ethnicities"
    print(prompt)

    
    for i in range(args.num_imgs):
       
        image = pipe(
            prompt.replace("\n", "").replace("$", "").replace("'", ""),
        ).images[0]


        name = prompt.replace(" ", "_").replace("\n", "").replace("$", "")
        os.makedirs(
            os.path.join(
                folder, prompt.replace("_", " ").replace("\n", "").replace("$", "")
            ),
            exist_ok=True,
        )
        image.save(
            os.path.join(
                folder,
                prompt.replace("_", " ").replace("\n", "").replace("$", ""),
                f"{name}_{i}.png",
            )
        )

        times.to_csv(timefile)
