import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_HOME'] = '/data/zhujinguo/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/data/zhujinguo/hf_home'


import torch
from diffusers import VQDiffusionPipeline

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

output = pipeline("A house with a bike", truncation_rate=1.0)

image = output.images[0]

image.save("./teddy_bear.png")