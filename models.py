from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from joblib import dump, load
import torch
import pandas as pd


class BaseModel:
    """
    Abstract base class for models to ensure common interface.
    """
    def train(self):
        raise NotImplementedError


class LogisticModel(BaseModel):
    def __init__(self, dataset, text_column="post", target_column="target", test_size=0.3, random_state=42):
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
        self.model = LogisticRegression()

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset[self.text_column],
            self.dataset[self.target_column],
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        self.vectorizer.fit(X_train)
        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        print("Training completed for Logistic Model.")
        return self.model, self.vectorizer


class DistilBERTModel(BaseModel):
    def __init__(self, dataset, text_column="post", target_column="target", max_length=512, output_dir="./results"):
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.max_length = max_length
        self.output_dir = output_dir
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    def train(self):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.dataset[self.text_column].tolist(),
            self.dataset[self.target_column].tolist(),
            test_size=0.2,
            random_state=42
        )

        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        val_encodings = self.tokenizer(
            val_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )

        train_dataset = torch.utils.data.TensorDataset(
            train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
        )

        print("Training completed for DistilBERT Model.")
        return self.model, self.tokenizer, val_dataset, val_labels


if __name__ == "__main__":
    # Load datasets
    birth_year_dataset = pd.read_parquet("datasets/birth_year_tokenized_cleaned.parquet")
    political_leaning_dataset = pd.read_parquet("datasets/political_leaning_tokenized_cleaned.parquet")

    # Initialize and train DistilBERT model for birth_year dataset
    print("Training DistilBERT Model on birth_year dataset...")
    birth_year_model = DistilBERTModel(
        dataset=birth_year_dataset,
        text_column="post",
        target_column="birth_year"
    )
    model, tokenizer, val_dataset, val_labels = birth_year_model.train()

    # Evaluate model
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch[2].cpu().numpy())

    print("Classification Report for birth_year dataset:")
    print(classification_report(true_labels, predictions))
    print(f"F1 Score: {f1_score(true_labels, predictions, average='weighted')}")

    # Initialize and train DistilBERT model for political_leaning dataset
    print("Training DistilBERT Model on political_leaning dataset...")
    political_leaning_model = DistilBERTModel(
        dataset=political_leaning_dataset,
        text_column="post",
        target_column="political_leaning"
    )
    model, tokenizer, val_dataset, val_labels = political_leaning_model.train()

    # Evaluate model
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch[2].cpu().numpy())

    print("Classification Report for political_leaning dataset:")
    print(classification_report(true_labels, predictions))
    print(f"F1 Score: {f1_score(true_labels, predictions, average='weighted')}")
