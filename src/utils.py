import json
import base64
import PIL
from io import BytesIO


# Step 1: Define default JSON template
default_json = {
    "mode":"txt2img",
    "prompt":"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "negative_prompt":"",
    "guidance_scale":2.5,
    "aspect_ratio":"1:1",
    "num_inference_steps":32,
    "init_image":None

}

aspect_ratios={

               "1:1":[1024,1024],
               "9:16":[720,1280],
               "2:3":[836,1254],
               "3:4":[876,1168],
               "4:3":[1168,876],
               "3:2":[1254,836],
               "16:9":[1280,720]

}

# Step 2: Function to update JSON based on another JSON
def update_json(base_json, update_json):
    for key, value in update_json.items():
        if key in base_json:
            base_json[key] = value
    return base_json


def decode(base64_image):
    image_bytes = base64.b64decode(base64_image)
    image = PIL.Image.open(BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image