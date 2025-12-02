# AI Crisis Sentiment & Emotion Analyzer

## Purpose
This repository contains a Streamlit application designed for sentiment and emotion analysis, specifically tailored for crisis communication scenarios (e.g., the Samsung Galaxy Note 7 crisis). It leverages advanced NLP models like BERTweet and DistilRoBERTa to classify text into sentiment categories (Positive, Neutral, Negative) and emotion categories (Joy, Anger, Fear, Sadness, Surprise, No Emotion). It also provides explainable AI (XAI) features to interpret model predictions.

## Features
- **Multi-Model Support**: Switch between BERTweet (optimized for social media) and DistilRoBERTa (general purpose).
- **Analysis Modes**:
    - **Single-task**: Uses separate models for sentiment and emotion.
    - **Multi-task**: Uses a joint model for both tasks.
- **Explainability**:
    - Token-level attribution scores (Integrated Gradients or Gradient Saliency).
    - Counterfactual analysis (what happens if a word is removed).
    - Interactive visualizations (heatmaps, probability charts).

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you want to use the interactive charts, ensure `plotly` is installed.*

## Models
The application expects pre-trained models to be placed in a `models/` directory in the root of the repository. This directory is not included in the repository due to file size limits.

You need to organize your models as follows:
```
models/
├── sentiment/
│   ├── bertweet/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer.json
│   └── distilroberta/
│       └── ...
├── emotion/
│   ├── bertweet/
│   │   └── ...
│   └── distilroberta/
│       └── ...
└── mtl/
    ├── bertweet/
    │   └── ...
    ├── distilroberta/
    │   └── ...
    ├── mtl_model.pt
    └── base_tok/
```

## Usage

To run the application, execute the following command in the terminal:

```bash
streamlit run app.py
```

The application will open in your default web browser.

### Using the App
1. **Configuration**: Use the sidebar to select the "Analysis Mode" (Single-task vs Multi-task) and "AI Backbone" (BERTweet vs DistilRoBERTa).
2. **Input**: Enter text in the text area or select a sample text.
3. **Analyze**: Click the "Analyze Text" button.
4. **Results**: View the sentiment and emotion predictions, confidence scores, and detailed explanations.

## Directory Structure
- `app.py`: Main Streamlit application source code.
- `data/`: Directory for data files (e.g., Reddit posts).
- `enc/`: Label encoders (pickle files).
- `requirements.txt`: Python dependencies.
- `models/`: (External) Directory for pre-trained models.
