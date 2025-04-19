import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from IPython.display import display
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--style', choices=['general', 'se'])
args = parser.parse_args()

path = f'manual_inspection/{args.model}/{args.style}'

df = pd.read_csv(f'{path}/img_list.csv', sep=';')

# df = pd.DataFrame(columns=['Image', 'Blip Labelling', 'Manual Labelling'])
#  Blip gender classification for each sd version

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# folder_path = 'Images/Replacements_gender/G2'

for filename in os.listdir(path): # for each image

    img_path = os.path.join(path, filename)
    print(img_path)
    # print(img_path)
    # checking if it is a file
    if os.path.isfile(img_path) and filename.endswith('.png'):

        raw_image = Image.open(img_path).convert('RGB')

        # Gender classification

        question = "Is the protagonist in this image a Male or Female?"
        inputs = processor(raw_image, question, return_tensors="pt")


        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        df.loc[df['prompt'] == filename.replace(' ','_').replace('\n',''), 'blip_gender'] = answer

        # Ethnicity classification

        question = "What is the ethnicity of this person?"
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
            answer = "white"
        elif str(answer).lower() == "indian" or str(answer).lower() == "asian":
           answer = "asian"
        elif (
            str(answer).lower() == "black"
            or str(answer).lower() == "african american"
            or str(answer).lower() == "african"
        ):
            answer = "black"
        elif str(answer).lower() == "arab" or str(answer).lower() == "middle eastern":
            answer = "arab"
        else:
            answer = "other"
        df.loc[df['prompt'] == filename.replace(' ','_').replace('\n',''), 'blip_ethnicity'] = answer
        

df.to_csv(f'{path}/blip_label.csv')
print("csv file saved!")