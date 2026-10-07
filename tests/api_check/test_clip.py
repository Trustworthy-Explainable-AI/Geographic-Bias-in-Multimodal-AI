import torch
import clip
from PIL import Image
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)   

url = "https://picsum.photos/id/1015/800/600"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

text = clip.tokenize(["a photo of a chair", "a photo of a kitchen"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print("Similarity scores:", similarity[0].tolist())