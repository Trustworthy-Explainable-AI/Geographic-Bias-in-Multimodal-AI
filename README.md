# Uncovering Geographic Bias in Multimodal AI

## 🧠 Project Brief

This project investigates **geographic and socioeconomic bias in multimodal AI systems** using the **Dollar Street dataset**.

We evaluate three models:
- CLIP (contrastive, local)
- Qwen-VL (open-source)
- Gemini (closed model)

The goal is to understand how model performance varies across:
- Regions
- Income levels
- Object categories

We analyze errors such as:
- Hallucination
- Abstention
- Prediction collapse

---

## ⚙️ Setup

### 1. Create and activate a virtual environment
Create:
```bash
python -m venv {name of your virtual environment} e.g
python -m venv trustAI
```

Activate:
- Windows:
```bash
trustAI\Scripts\activate
```
- Mac/Linux:
```bash
source trustAI/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment variables
Create a .env file:
```bash
HUGGINGFACE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```


Notes:
- This Readme will be updated as the project progresses