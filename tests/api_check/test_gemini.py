import os
import google.genai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env!")


client = genai.Client(api_key=GEMINI_API_KEY)

LOCAL_IMAGE_PATH = "../data/dollar_street/images/000001_01.jpg"

if not os.path.exists(LOCAL_IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found at:\n{LOCAL_IMAGE_PATH}\n\n"
        "Please update LOCAL_IMAGE_PATH to a real .jpg file inside your images/ folder.\n"
        "Tip: Open the images/ folder and copy the full path of any photo."
    )

image = Image.open(LOCAL_IMAGE_PATH).convert("RGB")

response = client.models.generate_content(model= "gemini-2.5-flash",
    contents=f"""Describe this household item in one sentence. Then guess the likely country/continent and income level based on its appearance and context. 
    Be specific about any cultural or geographic clues you see. {image}""")

print(response.text)