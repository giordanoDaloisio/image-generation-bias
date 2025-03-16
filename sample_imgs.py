import os
from random import sample
from shutil import copy, rmtree
import pandas as pd
import numpy as np


name_list = []



base_path = os.path.join('manual_inspection', 'se')
rmtree(base_path)
os.makedirs(base_path, exist_ok=True)


for folder in os.listdir('imgs'):
    if 'portrait_of_a_Software_Engineer' in folder:
        for file in os.listdir(os.path.join('imgs', folder)):
            print(file)
            name_list.append(file.replace(" ","_").replace("\n",""))

manual_imgs = sample(name_list, 89)

for folder in os.listdir('imgs'):
    if 'portrait_of_a_Software_Engineer' in folder:
        for file in os.listdir(os.path.join('imgs', folder)):
            if file.replace(" ","_").replace("\n","") in manual_imgs:
                print(file)
                copy(os.path.join('imgs', folder, file), os.path.join('manual_inspection', 'se', file))

pd.DataFrame({
    'prompt': manual_imgs,
}).to_csv(os.path.join('manual_inspection', 'se', 'img_list.csv'))