import os


os.environ['HF_HOME'] = '/data/zhujinguo/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/data/zhujinguo/hf_home'
from dataclasses import dataclass
import torch 
from PIL import Image

@dataclass
class InferenceConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-eval-128_withddimckpt'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = InferenceConfig()

# Defining the noise scheduler
from diffusers import DDPMPipeline

model_name = 'ddpm-butterflies-128'
pipeline = DDPMPipeline.from_pretrained(model_name).to("cuda")

import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "eval_show")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


evaluate(config, config.num_epochs, pipeline)

