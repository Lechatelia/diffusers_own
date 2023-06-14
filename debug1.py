import os

os.environ['HF_HOME'] = '/data/zhujinguo/hf_home'

from diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")

image = ddpm(num_inference_steps=25).images[0]
image.save("ddpm.png")