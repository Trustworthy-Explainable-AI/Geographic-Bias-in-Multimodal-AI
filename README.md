# Uncovering Geographic Bias in Multimodal AI

## Project Brief

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


## Setup

**Note: All experiments and set up are peculiar to Lightening AI's GPU environment. You might need to create a Virtual Environment, depending on your set up**

### 1. Clone the repo
```bash
git clone https://github.com/Trustworthy-Explainable-AI/Geographic-Bias-in-Multimodal-AI.git
cd Geographic-Bias-in-Multimodal-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment variables
Create a `.env` file in the project root:
```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
GEMINI_API_KEY=your_gemini_key
```

### 4. Download images
```bash
python download_images.py
```
This downloads the Dollar Street dataset from Kaggle and copies the 977 evaluation images to `data/images/`.



## Running Experiments

Smoke test using (30 images):
```bash
python -m src.run_inference --models clip --sample 30
```

Full run:
```bash
python -m src.run_inference --models clip
python -m src.run_inference --models qwen
python -m src.run_inference --models gemini
```

Run all models at once:
```bash
python -m src.run_inference --models clip qwen gemini
```

Results are saved to `results/` as CSVs — one predictions file and summary tables per model.


## Notes
- Qwen requires a GPU (L4 or better recommended)
- CLIP runs on CPU
- Gemini requires a valid `GEMINI_API_KEY`
- This README will be updated as the project progresses