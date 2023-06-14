import os

os.environ['HF_HOME'] = '/data/zhujinguo/hf_home'


from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
