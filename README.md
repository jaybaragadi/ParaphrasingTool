ParaphrasingTool – Indic Language Paraphrasing and Hyphen Prediction

This repository contains a lightweight, interactive tool for paraphrasing and segment (hyphen) prediction in Telugu, Hindi, and Bengali.
The system combines a Flask backend with fine-tuned NER models and an interactive HTML/JavaScript-based user interface that supports synonym replacement, transliteration, and segment reordering.

Overview

This tool enables:

Indic language hyphen/segment boundary prediction

Word-level synonym generation

Click-to-swap synonym replacement

Automatic and updated transliteration in real time

Drag-and-drop segment reordering

English translation hints for clarity

Multi-language support (Telugu, Hindi, Bengali)

The system is designed for linguistic experimentation, paraphrasing, and text transformation workflows across Indic languages.

Demo Video

A working demonstration video is included in the repository:

20251118-2310-24.6053599.mp4

Features
1. Indic Language Hyphen Prediction

NER-based hyphen prediction models trained using SimpleTransformers for:

Telugu

Hindi

Bengali

2. Synonym-Based Paraphrasing

Clicking a word opens a persistent tooltip showing synonyms.

Clicking any synonym replaces the word immediately.

Original words can be restored.

3. Real-Time Transliteration

The Romanized form updates automatically after each synonym replacement or structural change.

4. Segment Reordering

Using drag-and-drop, users can rearrange detected segments to form alternative paraphrased structures.

5. Translation Support

Uses Google Translate for lightweight, inline English translation hints.

Project Structure
ParaphrasingTool/
│
├── app.py                       # Flask backend: model inference and REST API
├── index.html                   # Frontend: UI for paraphrasing and interaction
│
├── MTP_Models.ipynb             # Complete training workflow (All languages)
├── MTP_Models_Telugu.ipynb      # Telugu-specific training experiments
│
├── requirements.txt             # Python dependencies
├── telugu_sentences.txt         # Sample dataset
│
├── Tool Iterations Code.docx    # Documentation of development iterations
├── 20251118-2310...mp4          # Working demo
│
├── .gitignore                   # Excludes heavy model directories
└── README.md

Installation and Setup
1. Clone the repository
git clone https://github.com/jaybaragadi/ParaphrasingTool.git
cd ParaphrasingTool

2. Install dependencies
pip install -r requirements.txt


Ensure the environment includes recent versions of:

Python 3.9+

PyTorch

simpletransformers

googletrans

3. Add model folders (required)

The heavy model directories are excluded via .gitignore:

best_model/
best_model_bengali/
best_model_hindi/


Place your trained model directories here before running the application.

4. Run the backend server
python app.py

5. Access the UI

Open the following address in your browser:

http://127.0.0.1:5000/

How the Interface Works
Word Interaction

Selecting a word opens a stable synonym tooltip.
Users can replace words by clicking their synonyms.

Transliteration

Romanized text updates automatically after each change.

Segment Reordering

Detected hyphen segments can be rearranged manually to produce paraphrased structures.

Model Information

Models were trained using the SimpleTransformers library on custom Indic datasets.
Key components include:

Transformer-based NER architecture

Segment boundary classification

Token-level evaluation metrics

Support for Hindi, Telugu, and Bengali models

Training experiments documented in Jupyter notebooks

Future plans include reinforcement learning–based paraphrase generation (EVL/PVL), extended synonym dictionaries, and model deployments.

Future Work

Reinforcement Learning–based paraphrasing (EVL / PVL)

Additional language coverage

Deployment via Streamlit or HuggingFace Spaces

Audio-to-text-to-paraphrase integration

Larger synonym datasets and contextual replacements

GPU-based inference for production setups

Contributing

Contributions are welcome.
Please open an issue or submit a pull request for improvements or bug fixes.
