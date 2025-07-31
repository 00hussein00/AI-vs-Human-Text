#  AI vs Human Text Classification

This project implements a full machine learning pipeline using deep learning techniques to classify whether a given text is **AI-generated** or **human-written**. It includes preprocessing, model training, evaluation, and an interactive UI for real-time predictions.

---

##  Project Overview

With the rise of large language models like GPT, distinguishing between AI-generated and human-authored content has become a growing challenge. This notebook-based project provides a practical solution using natural language processing (NLP) and deep learning.

### Key Features:
- Balanced dataset for fair training
- Advanced text preprocessing (stopwords & emoji removal)
- Deep learning models (LSTM, Bidirectional LSTM)
- Evaluation metrics and visualization
- Interactive user interface with live prediction

---

##  Contents

- `ai-vs-human-text.ipynb` — Jupyter Notebook containing the complete implementation
- `description.json` — Project metadata
- `README.md` — This file

---

##  Workflow

1. **Data Loading & Exploration**  
   Load and explore a dataset with labeled human and AI texts. Balance the dataset to avoid model bias.

2. **Text Preprocessing**  
   Clean text by removing noise such as stopwords and emojis.

3. **Vectorization**  
   Use TensorFlow's `TextVectorization` layer to prepare text for model input.

4. **Modeling**  
   Train various models (e.g., LSTM, BiLSTM) and compare their performance using accuracy/loss.

5. **Evaluation**  
   Visualize training history and make performance comparisons.

6. **Interactive UI**  
   Build a simple form-based UI with `ipywidgets` for real-time classification of custom text inputs.

---

##  Dependencies

Install the required libraries with:

```bash
pip install tensorflow pandas numpy matplotlib seaborn ipywidgets scikit-learn
```

ذ
---
تحرير
# Try a prediction
predict("I am Human.")  # Example of human-written text
You can also use the interactive input box at the bottom of the notebook.

 # Models Used
**LSTM (Long Short-Term Memory)**

**Bidirectional LSTM**

**Baseline Dense models (optional)**


# Tools & Frameworks
**Python**

**Jupyter Notebook**

**TensorFlow / Keras**

**ipywidgets**

**scikit-learn**

**matplotlib & seaborn for visualization**

# Dataset
You can use any labeled dataset that contains samples of both human-written and AI-generated texts. In this project, a dataset sourced from Kaggle was used (you can customize this based on your dataset).

# License
This project is licensed under the MIT License.
