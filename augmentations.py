import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import string
import pandas as pd


model_name = "Vamsi/T5_Paraphrase_Paws"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def paraphrase(text, max_length=64):
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids = input_ids,
        attention_mask = attention_masks,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        temperature=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_augmentations(data, labels = ["surprise"]):
    aug_phrases = []
    aug_labels = []
    for label in labels:
        label_phrases = data[data["label"] == label]["text"]
        paraphrased_phrases = [paraphrase(phrase).lower().translate(str.maketrans('', '', string.punctuation)) for phrase in label_phrases]
        phrase_labels = [label]*len(paraphrased_phrases)
        aug_phrases.extend(paraphrased_phrases)
        aug_labels.extend(phrase_labels)
    aug_data = pd.DataFrame({
        "text": aug_phrases,
        "label": aug_labels
    })
    aug_data.to_csv("./data/train_aug.csv", header=False, index=False)
    return aug_data
