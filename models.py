from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models.fasttext import FastText
from config import __RANDOM_STATE__, test_size
from reader import Reader


class DataPreprocessor:
    """
    Vectorize features with either TF-IDF for logistic regression or FastText for SVM.
    """
    def __init__(self, reader: Reader.dataset, vectorizer, mode, text_column="post",
                 test_size=test_size, random_state=__RANDOM_STATE__):
       self.reader = reader
       self.text_column = text_column
       self.test_size = test_size
       self.random_state = random_state
       self.mode = mode
       self.vectorizer = vectorizer

    def mapping(self, y_train, y_test):
        if self.mode == "political_leaning":
            y_train = y_train.map({'left': 0, 'center': 1, 'right': 2})
            y_test = y_test.map({'left': 0, 'center': 1, 'right': 2})
        return y_train, y_test

    def preprocess(self):
        X_train, X_test, y_train, y_test = self.reader.split_data(self.test_size, self.random_state)
        y_train, y_test = self.mapping(y_train, y_test)

        print("Fitting the vectorizer...")
        self.vectorizer.fit(tqdm(X_train, desc="Vectorizing"))
        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        return X_train_vec, X_test_vec, y_train, y_test

# x_train_vec, x_test_vec, y_train, y_test = DataPreprocessor(Reader('datasets/birth_year.csv', tokenize=False),
#                                                             TfidfVectorizer(use_idf=True, max_df=0.95),
#                                                             'generation').preprocess()
# print(y_train)


class LogisticModel:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = preprocessor.vectorizer
        # self.vectorizer = vectorizer
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=preprocessor.random_state)
        # self.parameters_before_tuning = None
        # self.parameters_after_tuning = None

    def train(self):
        X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

        parameters_before_tuning = self.model.get_params()
        print("Parameters before tuning:", parameters_before_tuning)

        print("Training the Logistic Model...")
        self.model.fit(tqdm(X_train_vec, desc="Training Logistic Model"), y_train)
        print("Training completed.")

        return self.model, self.vectorizer

lr = LogisticModel(DataPreprocessor(Reader('datasets/birth_year.csv', tokenize=False),
                                                            TfidfVectorizer(use_idf=True, max_df=0.95),
                                                            'generation').preprocess())
print(lr)


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
