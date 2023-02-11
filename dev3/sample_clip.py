import numpy as np
import torch
from PIL import Image
import numpy as np
import torch
import clip

filename = "yaki-udon.jpg"

model, preprocess = clip.load("ViT-B/32")
image = Image.open(filename).convert("RGB")
image_input = torch.tensor(np.stack([preprocess(image)]))

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    print(image_features)
