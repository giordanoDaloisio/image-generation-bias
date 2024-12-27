import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from IPython.display import display


df = pd.DataFrame(columns=['Image', 'Blip Labelling', 'Manual Labelling'])
#  Blip gender classification for each sd version

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

folder_path = 'Images/Replacements_gender/G2'

for filename in os.listdir(folder_path): # for each image

    img_path = os.path.join(folder_path, filename)
    print(img_path)
    # print(img_path)
    # checking if it is a file
    if os.path.isfile(img_path):

        raw_image = Image.open(img_path).convert('RGB')

        # question1 = "Is this image of a human?"
        # inputs1 = processor(raw_image, question1, return_tensors="pt").to("cuda")


        # out1 = model.generate(**inputs1)
        # answer1 = processor.decode(out1[0], skip_special_tokens=True)

        # if answer1 == 'no':
        #     df.loc[len(df.index)] = [filename, "N", ''] 
        #     continue

        question = "Is the protagonist in this image a Male or Female?"
        inputs = processor(raw_image, question, return_tensors="pt")


        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        if answer == 'male':
            df.loc[len(df.index)] = [filename, "M", ''] 
        if answer == 'female':
            df.loc[len(df.index)] = [filename, "F", ''] 

df.to_csv('Stats/Replacements/G_gender_2.csv')
print("csv file saved!")