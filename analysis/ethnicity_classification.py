import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from IPython.display import display
from argparse import ArgumentParser
import shutil
import json

parser = ArgumentParser()
parser.add_argument("--model", default="3", choices=["2", "3", "xl", "flux", "segmind", "openjourney"])
parser.add_argument("--type", choices=["General", "SE"])

args = parser.parse_args()

df = pd.DataFrame(columns=["Prompt", "Asian", "White", "Black", "Arab", "Other"])
#  Blip gender classification for each sd version

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

if args.model == "2" or args.model == "3" or args.model == "xl":
    folder_path = f"../Images/{args.type}/imgs{args.stable_version}"
else:
    folder_path = f"../imgs_{args.model}/{args.type}"
other_list = {}

for folder in os.listdir(folder_path):
    # for folder in os.listdir(folder_path): # for each prompt
    asian_count = 0
    white_count = 0
    black_count = 0
    arab_count = 0
    other_count = 0
    folder_join = os.path.join(folder_path, folder)
    print(folder_path)
    if not os.path.isdir(folder_join):
        continue
    for filename in os.listdir(folder_join):  # for each image

        img_path = os.path.join(folder_join, filename)
        print(img_path)
        # print(img_path)
        # checking if it is a file
        if os.path.isfile(img_path):

            raw_image = Image.open(img_path).convert("RGB")
            question = "What is the ethnicity of the person in this image?"
            inputs = processor(raw_image, question, return_tensors="pt")

            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)
            # Asian, Black, White, Arab
            if (
                str(answer).lower() == "caucasian"
                or str(answer).lower() == "white"
                or str(answer).lower() == "italian"
                or str(answer).lower() == "german"
                or str(answer).lower() == "hispanic"
            ):
                white_count += 1
            elif str(answer).lower() == "indian" or str(answer).lower() == "asian":
                asian_count += 1
            elif (
                str(answer).lower() == "black"
                or str(answer).lower() == "african american"
                or str(answer).lower() == "african"
            ):
                black_count += 1
            elif (
                str(answer).lower() == "arab" or str(answer).lower() == "middle eastern"
            ):
                arab_count += 1
            else:
                other_count += 1
                if str(answer).lower() in other_list:
                    other_list[str(answer)] += 1
                else:
                    other_list[str(answer)] = 1

    print(
        f'Classifying "{folder}" done! White: {white_count},  Asian: {asian_count}, Black: {black_count}, Arab: {arab_count}, Other: {other_count}'
    )
    df.loc[len(df.index)] = [
        folder,
        asian_count,
        white_count,
        black_count,
        arab_count,
        other_count,
    ]

print(f"Other ethnicities identified: {other_list}")


if args.type == "General":
    prompt_type = "G"
else:
    prompt_type = "SE"

output_file = f"Stats/{prompt_type}_ethnicity_count_{args.model}.csv"

df.to_csv(output_file)
print("output file saved!")
display(df)


f = open("Stats/other_ethnicities.txt", "a")
others = json.dumps(other_list)
f.write(others)
f.close()
