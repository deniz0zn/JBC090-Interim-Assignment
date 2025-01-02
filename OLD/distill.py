import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import shap
import pandas as pd
from joblib import dump, load

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataPreprocessor:
    def __init__(self, dataset, text_column="post", target_column="target",
                 tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'),
                 max_length=512):
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self):
        texts = self.dataset[self.text_column].tolist()
        labels = self.dataset[self.target_column].tolist()
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = torch.tensor(labels)
        return encodings, labels

class DistilBERTModelTrainer:
    def __init__(self, preprocessor, model=None, output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16):
        self.preprocessor = preprocessor
        self.model = model if model else DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size

    def train(self):
        dataset = self.preprocessor.preprocess()
        train_size = int(0.8 * len(dataset))
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        dump(self.model, 'distilbert_model.pkl')
        dump(self.preprocessor.tokenizer, 'distilbert_tokenizer.pkl')
        print(f"Model and tokenizer saved to 'distilbert_model.pkl' and 'distilbert_tokenizer.pkl'")

class DistilBERTModelEvaluator:
    def __init__(self, model_path='distilbert_model.pkl', tokenizer_path='distilbert_tokenizer.pkl'):
        self.model = load(model_path)
        self.tokenizer = load(tokenizer_path)

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions

    def evaluate(self, texts, true_labels):
        predictions = self.predict(texts)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        return accuracy, report

    def explain(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        explainer = shap.Explainer(self.model, inputs)
        shap_values = explainer(inputs)
        shap.summary_plot(shap_values, features=inputs)
