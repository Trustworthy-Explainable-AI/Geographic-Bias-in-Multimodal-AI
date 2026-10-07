import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

url = "https://picsum.photos/id/1015/800/600"

try:
    response = client.chat.completions.create(
        model="qwen/qwen-2.5-vl-72b-instruct",
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

    print("Qwen's response:", response.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")