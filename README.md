# Analysis of Gender and Ethnicity Bias of Stable Diffusion Models Towards Software Engineering Tasks

This repository contains the data and analysis scripts for the paper _"How Do Generative Models Draw a Software Engineer? An Empirical Study on Implicit Bias of Open Source Image Generation Models"_.

Generated images are stored in the `Images` folder.

Scripts to perform the analysis reported in the paper are stored in the `analysis` folder.

To generate images using the different models and prompts, the `image_generation.py` script can be used. The script requires the `diffusers` library and a compatible GPU. The script can be run with the following command:

```bash
python image_generation.py --model <model_name> --num_images <number_of_images> --type <General or SE> <--fair>
```

Where `<model_name>` is the name of the model to be used, and `<number_of_images>` is the number of images to generate.

Required libraries can be installed through conda using the command:

```bash
conda env create -f environment.yml
conda activate sdenv
```
