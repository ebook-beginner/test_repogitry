import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import clip

model, preprocess = clip.load("ViT-B/32")
model.eval()

def feature(img_path):
    """画像を開き、埋め込みを計算して返す

    Args:
        img_path (str): 画像ファイルのパス

    Returns:
        torch.tensor: 埋め込み
    """
    with torch.no_grad():
        pil_img = Image.open(img_path).convert("RGB")
        image_input = torch.tensor(np.stack([preprocess(pil_img)]))
        image_features = model.encode_image(image_input).float()

    return image_features

rows = []

for root, dirs, files in os.walk(top='./data'):
    for file in files:
        file = os.path.join(root, file)
        
        if file.lower().endswith(('.jpeg', '.jpg', '.png')):
            print(file)
            buffer = []
            f = feature(file)
            rows.append([file] + f.tolist()[0])

df = pd.DataFrame(rows, columns=["filepath"] + [x for x in range(len(f.tolist()[0]))])
df.to_csv("feature.csv", encoding='utf_8_sig')


