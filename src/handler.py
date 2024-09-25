""" Example handler file. """

import runpod
import torch
import io
import time
from utils import *
from diffusers import DiffusionPipeline
from diffusers import EDMDPMSolverMultistepScheduler

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:

    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details

    pipe.scheduler = EDMDPMSolverMultistepScheduler()

except RuntimeError:
    quit()


def handler(job):
    """ Handler function that will be used to process jobs. """
    time_start = time.time()
    input_json = job['input']
    updated_json = update_json(default_json, input_json)
    mode = updated_json['mode']
    prompt = updated_json['prompt']
    negative_prompt = updated_json['negative_prompt']
    guidance_scale = updated_json['guidance_scale']
    num_inference_steps = updated_json['num_inference_steps']
    height, width = aspect_ratios[updated_json['aspect_ratio']]

    height = height - (height % 8)
    width = width - (width % 8)
    if mode == 'img2img':
        init_image = decode(updated_json['init_image'])
        image = \
        pipe(image=init_image, prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                     height=height, width=width, num_inference_steps=num_inference_steps, ).images[0]
    elif mode == 'txt2img':
        image = \
        pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, height=height,
                     width=width, num_inference_steps=64, ).images[0]

    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
