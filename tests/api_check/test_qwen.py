import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load env and setup client
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") # Or TOGETHER_API_KEY

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

# 2. Prepare the Image
url = "https://picsum.photos/id/1015/800/600"
# Qwen via API usually prefers a URL or a Base64 string
print(f"🔄 Requesting Qwen-VL via API...")

try:
    response = client.chat.completions.create(
        model="qwen/qwen-2.5-vl-72b-instruct", # Using the 72B version for better bias research!
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this scene and guess which country it might be from."},
                    {
                        "type": "image_url",
                        "image_url": {"url": url}
                    },
                ],
            }
        ],
    )

    print("\n✅ SUCCESS: Qwen-VL (Remote) is working!")
    print("-" * 30)
    print("Qwen's response:", response.choices[0].message.content)

except Exception as e:
    print(f"❌ Error: {e}")