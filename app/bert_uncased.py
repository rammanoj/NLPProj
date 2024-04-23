from torch import tensor, cuda, device
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, BertTokenizer, BertForSequenceClassification
from torch.optim import RMSprop
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import torch


# Class to load the dataset for training the model
class ABSADataset(Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_length=128):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, aspect,
                                  return_tensors='pt',
                                  max_length=self.max_length,
                                  truncation=True,
                                  padding='max_length')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': tensor(label)
        }

  
class CrossEntropyLossFunc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


class BERTUncased:

    def __init__(self, num_epochs) -> None:
        self.model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained("bert_based_config.json")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, config=config)

        # 4 labels: positive, negative, neutral and conflict
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=4)
        self.optimizer = RMSprop(self.model.parameters(), lr=3e-5)
        self.num_epochs = num_epochs
        self.batch_size = 4

        if cuda.is_available():
            self.device = device("cuda")
        else:
            self.device = device("cpu")

        self.model.to(self.device)
        self.aspect_numb = {"negative": 0, "positive": 1, "neutral": 2, "conflict": 3}

    
    def get_dataset(self, dataset):
        input_text, labels, aspects = [], [], []
        dataset = pd.DataFrame(dataset)
        for i, r in dataset.iterrows():
            input_text.append(r["input_text"])
            aspects.append(r["aspect"])
            labels.append(self.aspect_numb[r["polarity"]])
        return input_text, labels, aspects

    def train(self, dataset):
        input_text, labels, aspects = self.get_dataset(dataset)

        # Create training dataset and the dataloader
        absa_data = ABSADataset(input_text, aspects, labels, self.tokenizer)
        dataloader = DataLoader(absa_data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            self.model.train()
            loss_data = []
            print("Starting Epoch", epoch + 1)
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = CrossEntropyLossFunc()(outputs.logits, labels)
                loss_data.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print("Loss: ", np.mean(loss_data))


    def test(self, dataset):
        test_input_texts, test_labels, test_aspects = self.get_dataset(dataset)

        # absa_data testing dataset and the dataloader
        absa_data = ABSADataset(test_input_texts, test_aspects, test_labels, self.tokenizer)
        test_dataloader = DataLoader(absa_data, batch_size=self.batch_size, shuffle=False)

        # Evaluate model
        self.model.eval()
        predicted_labels, true_labels = [], []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Predict class with highest probability
                _, predicted = torch.max(logits, dim=1)

                # Move predictions and true labels to CPU for further processing
                predicted_labels.extend(predicted.tolist())
                true_labels.extend(labels.tolist())

        label_map = {0: 'negative', 1: 'positive', 2: 'neutral', 3: 'conflict'}
        # get classification report
        out_data = pd.DataFrame(dataset)
        out_data['y_pred'] = predicted_labels
        predicted_labels = [label_map[label] for label in predicted_labels]
        true_labels = [label_map[label] for label in true_labels]
        return out_data, classification_report(true_labels, predicted_labels)
