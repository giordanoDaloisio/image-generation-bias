import os
from together import Together
from requests import get
from argparse import ArgumentParser
from together_key import API_KEY
import time

parser = ArgumentParser()
parser.add_argument('--style')
args = parser.parse_args()

if(args.style == 'se'):
  file = 'prompts_se.txt'
else:
  file = 'prompts_g.txt'

client = Together(api_key=)

with open(file) as f:
  for prompt in f:
    path = os.path.join('imgs',prompt.replace(" ","_").replace("\n",""))
    os.makedirs(path, exist_ok=True)
    for i in range(20):
      response = client.images.generate(
          prompt=prompt,
          model="black-forest-labs/FLUX.1-schnell-Free",
          steps=1,
          n=1,
      )
      image = get(response.data[0].url).content
      with open(os.path.join(path, f"{prompt}_{i}.png"), "wb") as f:
        f.write(image)
      time.sleep(2)