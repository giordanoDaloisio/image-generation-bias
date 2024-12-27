import pandas as pd
import os
import shutil

general = open('folder_names_g.txt', 'r')
lines_g = general.readlines()
print(lines_g[0])
# count = 0

# folder = 'Images/SE/imgsxl'
prompt = open('prompts_g.txt', 'w')

for line in lines_g:
    newline = line.replace('_', ' ')
    print(newline)
    prompt.write(newline)
    # break
    # filename = os.path.join(folder, newline)
    # os.mkdir(filename)
    # count+=1
    # print (f"folder {count} created!")
    # # break


#     # # prompt.write(newline)
#     # print (line.replace('\n', '')).replace('$', '')
#     # print(newline)
#     # break
    
