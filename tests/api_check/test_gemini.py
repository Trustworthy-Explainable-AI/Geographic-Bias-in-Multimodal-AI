import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# ====================== LOAD .env ======================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env!")

print(f"🔑 API key loaded (starts with): {GEMINI_API_KEY[:8]}...")
genai.configure(api_key=GEMINI_API_KEY)
# ======================================================

print("🔄 Loading Gemini 2.5 Flash...")
model = genai.GenerativeModel("gemini-2.5-flash")

# ====================== LOCAL DOLLAR STREET IMAGE ======================
# ← CHANGE THIS PATH to any real image from your downloaded dataset
LOCAL_IMAGE_PATH = "../data/dollar_street/images/000001_01.jpg"   # ← example placeholder

if not os.path.exists(LOCAL_IMAGE_PATH):
    raise FileNotFoundError(
        f"❌ Image not found at:\n{LOCAL_IMAGE_PATH}\n\n"
        "Please update LOCAL_IMAGE_PATH to a real .jpg file inside your images/ folder.\n"
        "Tip: Open the images/ folder and copy the full path of any photo."
    )

image = Image.open(LOCAL_IMAGE_PATH).convert("RGB")
print(f"✅ Loaded local Dollar Street image: {LOCAL_IMAGE_PATH}")
# =======================================================================

print("🔄 Running inference on real Dollar Street household image...")
response = model.generate_content([
    "Describe this household item in one sentence. "
    "Then guess the likely country/continent and income level based on its appearance and context. "
    "Be specific about any cultural or geographic clues you see.",
    image
])

print("\n✅ SUCCESS: Gemini is working with REAL local Dollar Street data!")
print("Gemini's response:\n")
print(response.text)