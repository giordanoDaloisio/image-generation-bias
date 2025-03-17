import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from IPython.display import display
from argparse import ArgumentParser
import shutil

parser = ArgumentParser()
parser.add_argument("--model", default="3", choices=["2", "3", "xl", "flux"])
parser.add_argument("--type", choices=["General", "SE"])

args = parser.parse_args()

df = pd.DataFrame(columns=["Prompt", "Male", "Female"])
#  Blip gender classification for each sd version

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

if args.model == "2" or args.model == "3" or args.model == "xl":
    folder_path = f"../Images/{args.type}/imgs{args.stable_version}"
else:
    folder_path = f"../imgs_{args.model}/{args.type}"

for folder in os.listdir(folder_path):
    # for folder in os.listdir(folder_path): # for each prompt
    male_count = 0
    female_count = 0
    not_human_count = 0
    folder_join = os.path.join(folder_path, folder)
    print(folder_path)
    for filename in os.listdir(folder_join):  # for each image

        img_path = os.path.join(folder_join, filename)
        print(img_path)
        # print(img_path)
        # checking if it is a file
        if os.path.isfile(img_path):

            raw_image = Image.open(img_path).convert("RGB")
            question1 = "Is this image showing a human?"
            inputs1 = processor(raw_image, question1, return_tensors="pt")

            out1 = model.generate(**inputs1)
            answer1 = processor.decode(out1[0], skip_special_tokens=True)

            if (
                answer1 == "no"
            ):  # move to new folder based on inputs, incerement both male and femalae count, then continue
                shutil.move(
                    img_path, f"Images/Not_human/{args.type}_{args.stable_version}"
                )
                male_count += 1
                female_count += 1
                # not_human_count+=1
                continue

            question = "Is the person in this image a Male or a Female?"
            inputs = processor(raw_image, question, return_tensors="pt")

            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)

            if answer == "male":
                male_count += 1
            if answer == "female":
                female_count += 1

    print(
        f'Classifying "{folder}" done! Male: {male_count},  Female: {female_count}, Not Human: {not_human_count}'
    )
    df.loc[len(df.index)] = [folder, male_count, female_count]

if args.type == "General":
    prompt_type = "G"
else:
    prompt_type = "SE"

output_file = f"Stats/{prompt_type}_gender_count_{args.model}.csv"

df.to_csv(output_file)

display(df)
