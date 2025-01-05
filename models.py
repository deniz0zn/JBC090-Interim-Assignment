import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from gensim.models.fasttext import FastText
from config import __RANDOM_STATE__,test_size
from reader import Reader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from fine_tune import fine_tune_svm,fine_tune_log_reg
from sklearn.base import BaseEstimator
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification



class DataPreprocessor:
   def __init__(self, reader: Reader(), vectorizer, text_column="post", target_column="target",
                test_size=test_size, random_state=__RANDOM_STATE__, mode= "political_leaning",):
       self.reader = reader
       self.text_column = text_column
       self.target_column = target_column
       self.test_size = test_size
       self.random_state = random_state
       self.mode = mode
       self.vectorizer = vectorizer

   def mapping(self, y_train, y_test):
       if self.mode == "political_leaning":
           y_train = y_train.map({'left': 0, 'center': 1, 'right': 2})
           y_test = y_test.map({'left': 0, 'center': 1, 'right': 2})

       return y_train,y_test

   def preprocess(self):
       X_train, X_test, y_train, y_test = self.reader.split_data(self.test_size,self.random_state)
       y_train,y_test = self.mapping(y_train, y_test)

       print("Fitting the vectorizer...")
       self.vectorizer.fit(tqdm(X_train, desc="Vectorizing"))
       X_train_vec = self.vectorizer.transform(X_train)
       X_test_vec = self.vectorizer.transform(X_test)

       return X_train_vec, X_test_vec, y_train, y_test

class LogisticModel:
    def __init__(self, preprocessor: DataPreprocessor()):
        self.preprocessor = preprocessor
        self.vectorizer = preprocessor.vectorizer
        self.model = LogisticRegression()

        self.parameters_before_tuning = None
        self.parameters_after_tuning = None


    def train(self):
        X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

        self.parameters_before_tuning = self.model.get_params()
        print("Parameters before tuning:", self.parameters_before_tuning)

        print("Training the Logistic Model...")
        self.model.fit(tqdm(X_train_vec, desc="Training Logistic Model"), y_train)
        print("Training completed.")
        return self.model, self.vectorizer

    # def fine_tuning(self):
    #     best_model,vectorizer, parameters = fine_tune_svm(self.model,"fine_tuned/logistic/model.pkl",
    #                       self.vectorizer, "fine_tuned/logistic/vectorizer.pkl",
    #                       self.X_train_vec, self.y_train_vec)
    #
    #     self.parameters_after_tuning = parameters
    #     print("Parameters after tuning:", self.parameters_after_tuning)

        # return best_model,vectorizer, parameters


# class DistilBERTModel:
#     def _init_(self, dataset, text_column="post", target_column="target", max_length=512, output_dir="./results"):
#         self.dataset = dataset
#         self.text_column = text_column
#         self.target_column = target_column
#         self.max_length = max_length
#         self.output_dir = output_dir
#         self.unique_labels = sorted(self.dataset[self.target_column].unique())
#         self.label_to_id = {label: idx for idx, label in enumerate(self.unique_labels)}
#         self.id_to_label = {idx: label for idx, label in enumerate(self.unique_labels)}
#         self.num_labels = len(self.unique_labels)
#
#         self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#         self.model = DistilBertForSequenceClassification.from_pretrained(
#             'distilbert-base-uncased',
#             num_labels=self.num_labels
#         )
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#
#     def train(self):
#         # Map labels to IDs
#         self.dataset[self.target_column] = self.dataset[self.target_column].map(self.label_to_id)
#
#         train_texts, val_texts, train_labels, val_labels = train_test_split(
#             self.dataset[self.text_column].tolist(),
#             self.dataset[self.target_column].tolist(),
#             test_size=0.2,
#             random_state=42
#         )
#
#         print("Tokenizing training and validation data...")
#         train_encodings = self.tokenizer(
#             train_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
#         )
#         val_encodings = self.tokenizer(
#             val_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
#         )
#
#         train_dataset = torch.utils.data.TensorDataset(
#             train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
#         )
#         val_dataset = torch.utils.data.TensorDataset(
#             val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
#         )
#
#         print("Training DistilBERT model...")
#         train_loader = torch.utils.data.Dataset(train_dataset, batch_size=32, shuffle=True)
#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
#
#         self.model.train()
#         for epoch in range(3):
#             loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
#             for batch in loop:
#                 inputs = {
#                     "input_ids": batch[0].to(self.device),
#                     "attention_mask": batch[1].to(self.device),
#                     "labels": batch[2].to(self.device),
#                 }
#                 optimizer.zero_grad()
#                 outputs = self.model(**inputs)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#
#                 # Update progress bar with loss
#                 loop.set_postfix(loss=loss.item())
#
#         print("Training completed for DistilBERT Model.")
#         return self.model, self.tokenizer, val_dataset,val_labels

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

   # def fine_tuning(self, model_path = "fine_tuned/logistic/model.pkl",
   #              vectorizer_path = "fine_tuned/logistic/vectorizer.pkl"):
   #     self.model, parameters = fine_tune(self.model,model_path,
   #                                     self.vectorizer, vectorizer_path,
   #                                     self.X_train_vec, self.y_train_vec)
   #     self.parameters_after_tuning = parameters
   #     print("Parameters after tuning:", self.parameters_after_tuning)
   #     # return best_model, parameters


