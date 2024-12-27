import os
import random
import shutil

source  = 'Images/SE/imgsxl'
dest = 'Testing/Images/SE/imgsxl'
no_of_files = 5

for i in range(no_of_files):
    random.seed(random.randint(0, 1000000000000))
    random_prompt = random.choice(os.listdir(source))
    random_file=random.choice(os.listdir(os.path.join(source, random_prompt)))
    source_file = os.path.join(source, random_prompt, random_file)
    dest_file = os.path.join(dest, random_file)
    while(os.path.exists(dest_file)): # no repeats
        random_prompt = random.choice(os.listdir(source))
        random_file=random.choice(os.listdir(os.path.join(source, random_prompt)))
        source_file = os.path.join(source, random_prompt, random_file)
        print(f"repeat {i}, {source_file} ")
    dest_file = os.path.join(dest, random_file)
    shutil.copy(source_file, dest_file)

print("successfully copied")

