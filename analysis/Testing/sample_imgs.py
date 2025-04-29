import os
from random import sample
from shutil import copy, rmtree
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model')

args = parser.parse_args()

name_list = []



base_path = os.path.join('manual_inspection', args.model)
# rmtree(base_path)
os.makedirs(base_path, exist_ok=True)

img_path = os.path.join(args.model, 'SE')

for folder in os.listdir(img_path):
    if folder == '.DS_Store':
        continue
    for file in os.listdir(os.path.join(img_path, folder)):
        print(file)
        name_list.append(file.replace(" ","_").replace("\n",""))

manual_imgs = sample(name_list, 89)

for folder in os.listdir(img_path):
    if folder == '.DS_Store':
        continue
    for file in os.listdir(os.path.join(img_path, folder)):
        if file.replace(" ","_").replace("\n","") in manual_imgs:
            print(file)
            copy(os.path.join(img_path, folder, file), os.path.join(base_path, file))

pd.DataFrame({
    'prompt': manual_imgs,
}).to_csv(os.path.join(base_path, 'img_list.csv'))