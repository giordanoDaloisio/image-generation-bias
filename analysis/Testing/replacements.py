import os
import random
import shutil

source  = 'Images/SE/imgs2'
original_dest = 'Testing/Images/SE/imgs2'
dest = 'Testing/Images/Replacements_skintone/SE2'

no_of_files = 5

for i in range(no_of_files):
    random.seed(random.randint(0, 1000000000000))
    random_prompt = random.choice(os.listdir(source))
    random_file=random.choice(os.listdir(os.path.join(source, random_prompt)))
    source_file = os.path.join(source, random_prompt, random_file)
    original_dest_file = os.path.join(original_dest, random_file)
    new_dest_file = os.path.join(dest, random_file)

    while(os.path.exists(original_dest_file) or os.path.exists(new_dest_file) ): # no repeats
        random_prompt = random.choice(os.listdir(source))
        random_file=random.choice(os.listdir(os.path.join(source, random_prompt)))
        source_file = os.path.join(source, random_prompt, random_file)
        print(f"repeat {i}, {source_file} ")
    new_dest_file = os.path.join(dest, random_file)
    shutil.copy(source_file, new_dest_file)

print("successfully copied")
