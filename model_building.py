from transformers import DebertaTokenizer, DebertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score
import torch
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os


class EmotionDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_and_train_model(train_data, val_data, label_mapping):
    max_length = 128
    batch_size = 8
    epochs = 20
    learning_rate = 2e-6

    train_data["label"] = train_data["label"].map(label_mapping)
    val_data["label"] = val_data["label"].map(label_mapping)

    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    # Create datasets
    train_dataset = EmotionDetectionDataset(
        train_data["text"].tolist(), train_data["label"].tolist(), tokenizer, max_length
    )
    val_dataset = EmotionDetectionDataset(
        val_data["text"].tolist(), val_data["label"].tolist(), tokenizer, max_length
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(label_mapping)), y=train_data["label"].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Model
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=len(label_mapping))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # Scheduler
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    
    # Loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Track best model
    best_val_loss = float("inf")
    best_model_state = None

    # Early stopping
    early_stopping_patience = 5
    no_improvement = 0
    
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_accuracy += accuracy_score(labels.cpu(), preds.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy /= len(val_loader)
        scheduler.step()
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        torch.cuda.empty_cache()

    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss.")

    # Save training losses
    loss_df = pd.DataFrame({"epoch": list(range(1, len(train_losses) + 1)), 
                        "train_loss": train_losses, 
                        "val_loss": val_losses})
    
    results_path = "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    loss_df.to_csv("results/training_losses.csv", index=False)
    
    return model, tokenizer
