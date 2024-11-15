from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mydataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]['text'], target_map[self.data.iloc[idx]['label']]

train_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/train.csv')
valid_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/valid.csv')
test_data = pd.read_csv('./nusax-main/datasets/sentiment/indonesian/test.csv')

target_map={
    'positive':1,
    'negative':0,
    'neutral':2
}

tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")

def collate_fn(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs



train_dataset = Mydataset(train_data)
valid_dataset = Mydataset(valid_data)
test_dataset = Mydataset(test_data)

trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn = collate_fn)
validloader = DataLoader(valid_dataset, batch_size=8, shuffle=True, collate_fn = collate_fn)


model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def evaluate():
    model.eval()
    acc_sum = 0
    with torch.inference_mode():
        for i, batch in enumerate(validloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=1)
            acc_sum += (pred.long() == batch["labels"].long()).float().sum()
        return acc_sum / len(valid_dataset)

def train(epochs = 1, log_step = 20):
    best_acc = 0
    os.makedirs("./senti_ckpt", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if (i+1) % log_step == 0:
                print(f"Epoch: {epoch}, Step: {i+1}, Loss: {loss.item()}")
        acc = evaluate()
        print(f"Epoch: {epoch+1}, Accuracy: {acc}")
        # save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./senti_ckpt/best_model.pth")

# train(epochs=10)

model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=3).to(device)
model.load_state_dict(torch.load("./senti_ckpt/best_model.pth",weights_only=True))

# example usage
sen = "Memang lemah kita ini aku juga penakut sebenarnya"
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    label_map = {v: k for k, v in target_map.items()}
    print(f"input: {sen}, prediction: {label_map[pred.item()]}")
    
# test
def test():
    correct_predictions = 0
    total_predictions = 0

    for i, dt in enumerate(test_dataset):
        with torch.inference_mode():
            if i==1:
                print(dt)
            inputs = tokenizer(dt[0], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            if i==1:
                print(logits)
            pred = torch.argmax(logits, dim=-1)
            label_map = {v: k for k, v in target_map.items()}
            if pred.item() == dt[1]:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy}")

def confusion_matrix():
    confusion_matrix = torch.zeros(3, 3)
    for i, dt in enumerate(test_dataset):
        with torch.inference_mode():
            inputs = tokenizer(dt[0], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1)
            confusion_matrix[pred.item(), dt[1]] += 1
    print(confusion_matrix)

test()
confusion_matrix()
