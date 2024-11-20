from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
from Visualization import get_confusion_matrix, plot_loss
from pathlib import Path

class Mydataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]['text'], target_map[self.data.iloc[idx]['label']]

def collate_fn(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

def evaluate():
    model.eval()
    acc_sum = 0
    total_loss = 0
    with torch.inference_mode():
        for i, batch in enumerate(validloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            pred = torch.argmax(outputs.logits, dim=1)
            acc_sum += (pred.long() == batch["labels"].long()).float().sum()
            total_loss += loss.item()
    acc = acc_sum / len(valid_dataset)
    avg_valid_loss = total_loss / len(validloader)
    return acc, avg_valid_loss

def train(num_epochs):
    best_acc = 0
    train_losses = []
    valid_losses = []
    os.makedirs("./senti_ckpt", exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        accuracy, avg_valid_loss = evaluate()
        valid_losses.append(avg_valid_loss)
        # save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "./senti_ckpt/best_model.pth")
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}")
        print(f"Validation Loss: {avg_valid_loss}, Validation Accuracy: {accuracy}")
    return train_losses, valid_losses
# test
def test():
    model.eval()
    all_labels = []
    all_preds = []
    correct_predictions = 0
    total_predictions = 0

    for i, dt in enumerate(test_dataset):
        with torch.inference_mode():
            # if i==1:
            #     print(dt)
            inputs = tokenizer(dt[0], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            # if i==1:
            #     print(logits)
            pred = torch.argmax(logits, dim=-1)
            all_labels.append(dt[1])
            all_preds.append(pred.item())
            if pred.item() == dt[1]:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")
    return all_labels, all_preds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/train.csv')
    valid_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/valid.csv')
    test_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/test.csv')

    target_map={
        'positive':1,
        'negative':0,
        'neutral':2
    }

    tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")
    model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_dataset = Mydataset(train_data)
    valid_dataset = Mydataset(valid_data)
    test_dataset = Mydataset(test_data)

    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn = collate_fn)
    validloader = DataLoader(valid_dataset, batch_size=8, shuffle=True, collate_fn = collate_fn)
    num_epochs = 10
    train_losses, valid_losses = train(num_epochs)
    model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=3).to(device)
    model.load_state_dict(torch.load("./senti_ckpt/best_model.pth",weights_only=True))
    all_labels, all_preds = test()
    map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    all_labels = [map[label] for label in all_labels]
    all_preds = [map[pred] for pred in all_preds]
    get_confusion_matrix(all_labels, all_preds, save_path=Path('./senti_ckpt'))
    plot_loss(train_losses, valid_losses, save_path=Path('./senti_ckpt'))
    # example usage
    # sen = "Memang lemah kita ini aku juga penakut sebenarnya"
    # with torch.inference_mode():
    #     inputs = tokenizer(sen, return_tensors="pt")
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
    #     logits = model(**inputs).logits
    #     pred = torch.argmax(logits, dim=-1)
    #     label_map = {v: k for k, v in target_map.items()}
    #     print(f"input: {sen}, prediction: {label_map[pred.item()]}")
