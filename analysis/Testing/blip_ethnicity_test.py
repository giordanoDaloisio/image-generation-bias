import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from IPython.display import display


df = pd.DataFrame(columns=["Image", "Blip Labelling", "Manual Labelling"])
#  Blip gender classification for each sd version

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

folder_path = "Images/SE/imgsxl"

for filename in os.listdir(folder_path):  # for each image

    img_path = os.path.join(folder_path, filename)
    print(img_path)
    # print(img_path)
    # checking if it is a file
    if os.path.isfile(img_path) and filename.endswith(".png"):

        raw_image = Image.open(img_path).convert("RGB")

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
            df.loc[len(df.index)] = [filename, "white", ""]
        elif str(answer).lower() == "indian" or str(answer).lower() == "asian":
            df.loc[len(df.index)] = [filename, "asian", ""]
        elif (
            str(answer).lower() == "black"
            or str(answer).lower() == "african american"
            or str(answer).lower() == "african"
        ):
            df.loc[len(df.index)] = [filename, "black", ""]
        elif str(answer).lower() == "arab" or str(answer).lower() == "middle eastern":
            df.loc[len(df.index)] = [filename, "arab", ""]
        else:
            df.loc[len(df.index)] = [filename, "other", ""]


df.to_csv("Stats/ehtnicity_labellings/SE_xl_ethnicities.csv")
print("csv file saved!")

print(df["Blip Labelling"].value_counts(dropna=False, ascending=True))
