import clip
import json
import requests
from PIL import Image
from io import BytesIO

with open('data/annotations/instances_val2017.json', 'r') as file:
    # Parse the JSON file into a Python object
    data = json.load(file)

# print(data["images"][0]["coco_url"])
    
images = data["images"]

def fetch_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# print(fetch_image(images[0]["coco_url"]).size)
# print(fetch_image(images[0]["coco_url"]).mode)
# fetch_image(images[0]["coco_url"]).show()

model, preprocess = clip.load("ViT-B/32", device="cpu")

