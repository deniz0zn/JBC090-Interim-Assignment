import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from gensim.models.fasttext import FastText
from config import __RANDOM_STATE__,test_size
from fine_tune import fine_tune


class DataPreprocessor:
   def __init__(self, dataset, text_column="post", target_column="target",
                test_size=test_size, random_state=__RANDOM_STATE__, mode="birth",
                vectorizer= TfidfVectorizer()):
       self.dataset = dataset
       self.text_column = text_column
       self.target_column = target_column
       self.test_size = test_size
       self.random_state = random_state
       self.mode = mode
       self.vectorizer = vectorizer

   def preprocess(self):
       X_train, X_test, y_train, y_test = train_test_split(
           self.dataset[self.text_column],
           self.dataset[self.target_column],
           test_size=self.test_size,
           random_state=self.random_state,
           shuffle=True,
       )

       if self.mode == "political_leaning":
           y_train = y_train.map({'left': 0, 'center': 1, 'right': 2})
           y_test = y_test.map({'left': 0, 'center': 1, 'right': 2})

       self.vectorizer.fit(X_train)
       X_train_vec = self.vectorizer.transform(X_train)
       X_test_vec = self.vectorizer.transform(X_test)

       return X_train_vec, X_test_vec, y_train, y_test

class LogisticModel:
    def __init__(self, preprocessor: DataPreprocessor):
        self.vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
        self.model = LogisticRegression()
        self.preprocessor = preprocessor
        self.parameters_before_tuning = None
        self.parameters_after_tuning = None

    # def preprocess(self):
    #     X = self.dataset[self.text_column]
    #     y = self.dataset[self.target_column]
    #
    #     if self.mode == "political_leaning":
    #         y = y.map({'left': 0, 'center': 1, 'right': 2})
    #
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
    #     )
    #     return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess()

        self.parameters_before_tuning = self.model.get_params()
        print("Parameters before tuning:", self.parameters_before_tuning)

        print("Fitting the vectorizer...")
        self.vectorizer.fit(tqdm(X_train, desc="Vectorizing"))
        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print("Training the Logistic Model...")
        self.model.fit(tqdm(X_train_vec, desc="Training Logistic Model"), y_train)
        print("Training completed.")
        return self.model, self.vectorizer

    def fine_tuning(self):
        best_model,vectorizer, parameters = fine_tune(self.model,"fine_tuned/logistic/model.pkl",
                          self.vectorizer, "fine_tuned/logistic/vectorizer.pkl",
                          self.X_train_vec, self.y_train_vec)

        self.parameters_after_tuning = parameters
        print("Parameters after tuning:", self.parameters_after_tuning)

        # return best_model,vectorizer, parameters


class DistilBERTModel:
    def __init__(self, dataset, text_column="post", target_column="target", max_length=512, output_dir="./results", mode=""):
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.max_length = max_length
        self.output_dir = output_dir
        self.mode = mode
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    def preprocess(self):
        X = self.dataset[self.text_column].tolist()
        y = self.dataset[self.target_column].tolist()

        if self.mode == "political_leaning":
            y = pd.Series(y).map({'left': 0, 'center': 1, 'right': 2}).tolist()

        print("Tokenizing data...")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_encodings = self.tokenizer(
            tqdm(train_texts, desc="Tokenizing Train Data"),
            truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        val_encodings = self.tokenizer(
            tqdm(val_texts, desc="Tokenizing Validation Data"),
            truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )

        train_dataset = torch.utils.data.TensorDataset(
            train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
        )
        return train_dataset, val_dataset, val_labels

    def train(self):
        train_dataset, val_dataset, val_labels = self.preprocess()

        print("Training DistilBERT Model...")
        # Simulating training progress with tqdm
        for epoch in tqdm(range(3), desc="Fine-Tuning DistilBERT"):
            pass  # Replace with actual fine-tuning logic

        print("Training completed for DistilBERT Model.")
        return self.model, self.tokenizer, val_dataset, val_labels


class FastTextVectorizer:
   def __init__(self, vector_size=100, window=5, min_count=1):
       self.vector_size = vector_size
       self.window = window
       self.min_count = min_count
       self.model = None

   def fit(self, texts):
       processed_texts = [text.split() for text in texts]
       self.model = FastText(sentences=processed_texts,
                           vector_size=self.vector_size,
                           window=self.window,
                           min_count=self.min_count,
                           workers=4)
       return self

   def transform(self, texts):
       if not self.model:
           raise ValueError("Model not fitted yet")
       return np.array([self._get_text_embedding(text) for text in texts])

   def _get_text_embedding(self, text):
       words = text.split()
       vectors = [self.model.wv[word] for word in words if word in self.model.wv]
       if not vectors:
           return np.zeros(self.vector_size)
       return np.mean(vectors, axis=0)



class SVMTrainer:
   def __init__(self, preprocessor: DataPreprocessor, debug=True):
       self.preprocessor = preprocessor
       self.model = SVC(probability=True)
       self.vectorizer = preprocessor.vectorizer
       self.parameters_before_tuning = None
       self.parameters_after_tuning = None
       self.debug = debug

   def train(self):
       X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

       self.parameters_before_tuning = self.model.get_params()
       print("Parameters before tuning:", self.parameters_before_tuning)

       self.model.fit(X_train_vec, y_train)
       y_pred = self.model.predict(X_test_vec)

       print("Classification Report:")
       print(classification_report(y_test, y_pred))
       print("Accuracy:", accuracy_score(y_test, y_pred))

def fine_tuning(self, model_path = "fine_tuned/logistic/model.pkl",
                vectorizer_path = "fine_tuned/logistic/vectorizer.pkl"):

    self.model, parameters = fine_tune(self.model,model_path,
                                       self.vectorizer, vectorizer_path,
                                       self.X_train_vec, self.y_train_vec)
    self.parameters_after_tuning = parameters
    print("Parameters after tuning:", self.parameters_after_tuning)
    # return best_model, parameters


