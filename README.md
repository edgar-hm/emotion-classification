# Emotion Classification Using DeBERTa

A deep learning NLP pipeline and desktop GUI for multi-class emotion classification across six categories (**joy**, **sadness**, **anger**, **fear**, **love**, and **surprise**)[cite: 2, 22]. Built by fine-tuning `microsoft/deberta-base` (`DebertaForSequenceClassification`) on 20,000 labeled phrases from the [Kaggle Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data) [cite: 10, 22, 24, 32].

📄 **Deep-Dive Documentation:** See the [Project Report (`report.pdf`)](./report.pdf) and [Presentation Slides (`presentation.pdf`)](./presentation.pdf) for full exploratory data analysis, training curves, T5 paraphrasing augmentation experiments, and error analysis[cite: 1, 20].

---

## Key Results

Evaluated on the 2,000-sample holdout test set[cite: 10, 22, 27]:

* **Overall Accuracy:** **93.85%** (Weighted F1-Score: **0.94** | Macro F1-Score: **0.91**)[cite: 14, 27]
* **Top-Performing Classes:** Sadness (`F1: 0.97`), Joy (`F1: 0.96`), and Anger (`F1: 0.94`)[cite: 13, 27]
* **Minority Classes:** Fear (`F1: 0.89`), Love (`F1: 0.88`), and Surprise (`F1: 0.84`)[cite: 13, 27]

---

## Repository Structure

```text
├── main.py                # End-to-end orchestration (EDA, training, evaluation, UI)
├── visualizations.py      # Class distribution, phrase length, and word frequency plots
├── augmentations.py       # T5-based paraphrasing augmentation for minority classes
├── model_building.py      # PyTorch dataset, class-weighted training loop, early stopping
├── evaluate.py            # Classification report, confusion matrix, and loss curve export
├── ui.py                  # Tkinter GUI supporting text input and Google Speech-to-Text
├── data/                  # Pre-split train (16k), test (2k), eval (2k), and stopword list
├── figures/               # Saved EDA and evaluation plots
├── results/               # Exported loss histories and evaluation metrics
├── report.pdf             # Full technical report
└── presentation.pdf       # Project summary slides