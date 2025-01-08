from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, log_loss, \
    matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models.fasttext import FastText
from config import __RANDOM_STATE__, test_size, __DEBUG__
from reader import Reader


class Metrics:
    """
    Print model's performance metrics.
    :attributes: X_test, y_test, model
    """
    def __init__(self, X_test, y_test, model) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    def __str__(self) -> str:
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)

        ## ADD TRY EXCEPT FOR ROC SCORE
        return (f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}\n"
                f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.3f}\n"
                f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.3f}\n"
                f"F1: {f1_score(self.y_test, y_pred, average='weighted'):.3f}\n"
                f"Log loss: {log_loss(self.y_test, y_prob):.3f}\n"
                f"MCC: {matthews_corrcoef(self.y_test, y_pred):.3f}\n"
                f"AUROC: {roc_auc_score(self.y_test, y_prob, multi_class='ovo'):.3f}\n"
                f"Classification report:\n{classification_report(self.y_test, y_pred)}\n")


class DataPreprocessor:
    """
    Preprocess features for logistic regression using a vectorizer.
    :attributes: reader, vectorizer, mode, test_size, random_state
    """
    def __init__(self, reader: Reader.dataset, vectorizer, mode, test_size=test_size, random_state=__RANDOM_STATE__) -> None:
        """
        Initialize a DataPreprocessor class.
        :param reader: Reader object
        :param vectorizer: vectorizer to use
        :param mode: dataset for preprocessing
        :param test_size: size of test data
        :param random_state: seed for reproducibility
        """
        self.reader = reader
        self.test_size = test_size
        self.random_state = random_state
        self.mode = mode
        self.vectorizer = vectorizer

    def mapping(self, y_train, y_test):
        """
        Map target of type string to integer.
        """
        if self.mode == "political_leaning":
            y_train = y_train.map({'left': 0, 'center': 1, 'right': 2})
            y_test = y_test.map({'left': 0, 'center': 1, 'right': 2})
        return y_train, y_test

    def preprocess(self):
        """
        Fit vectorizer with features.
        """
        X_train, X_test, y_train, y_test = self.reader.split_data(self.test_size, self.random_state)
        y_train, y_test = self.mapping(y_train, y_test)

        print("Fitting the vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print("Vectorizer fitted.\n")

        return X_train_vec, X_test_vec, y_train, y_test


class LogisticModel:
    """
    Run logistic regression model with default parameters and balanced class weights.
    :attributes: preprocessor
    """
    def __init__(self, preprocessor: DataPreprocessor) -> None:
        """
        Initialize a LogisticModel class.
        :param preprocessor: DataPreprocessor to preprocess the data
        """
        self.preprocessor = preprocessor
        self.vectorizer = preprocessor.vectorizer
        self.model = LogisticRegression(class_weight='balanced',
                                        max_iter=1000,
                                        random_state=self.preprocessor.random_state
                                        )

    def fit(self):
        """
        Fit logistic regression model with default parameters and balanced class weights and print metrics.
        """
        X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

        parameters_before_tuning = self.model.get_params()
        print(f"Parameters before tuning: {parameters_before_tuning}\n")

        print("Training the default logistic regression model...")
        self.model.fit(X_test_vec, y_test)
        print(f"Default model fitted. Metrics: {Metrics(X_test_vec, y_test, self.model)}")

        return self.model, self.vectorizer


class FastTextVectorizer:
    """
    Preprocess features for SVM using FastText vectorizer.
    :attributes: texts, vector_size, window, min_count
    """
    def __init__(self, texts, vector_size=100, window=5, min_count=1) -> None:
        """
        Initialize a FastTextVectorizer class.
        :param texts: string to vectorize
        :param vector_size: dimensionality of vectors
        :param window: context window size
        :param min_count: ignore words of occurrences below count
        """
        self.texts = texts
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def create_model(self):
        """
        Create FastText embeddings model.
        """
        processed_texts = [text.split() for text in self.texts]
        self.model = FastText(sentences=processed_texts,
                              vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              workers=4
                              )
        return self.model

    def transform(self):
        """
        Fit FastText embeddings model with default parameters.
        """
        if not self.model:
            self.model = self.create_model()
        return np.array([self.get_text_embedding(text) for text in self.texts])

    def get_text_embedding(self, text):
        """
        Return text embeddings for feature X.
        """
        words = text.split()
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)


class SVMModel:
    """
    Run logistic regression model with default parameters and balanced class weights.
    """
    def __init__(self, reader: Reader, X, target: str, test_size=test_size, random_state=__RANDOM_STATE__) -> None:
        """
        Initialize an SVMModel class.
        :param X: vectorized features
        :param df: dataset to train and get target
        :param target: target to predict
        :param test_size: size of test data
        :param random_state: seed for reproducibility
        """
        self.reader = reader.dataset()
        self.X = X
        # self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = SVC(probability=True,
                         class_weight="balanced",
                         random_state=self.random_state
                         )

    def fit(self):
        """
        Fit SVM model with default parameters and print metrics.
        """
        X_train_vec, X_test_vec, y_train, y_test = train_test_split(self.X,
                                                                    self.reader[self.target],
                                                                    test_size=self.test_size,
                                                                    random_state=self.random_state
                                                                    )
        print(f"Parameters before tuning: {self.model.get_params()}\n")
        self.model.fit(X_train_vec, y_train)
        print(f"Default model fitted. Metrics: {Metrics(X_test_vec, y_test, self.model)}")

        return self.model


## LOGISTIC REGRESSION
# df_gen = Reader('datasets/birth_year.csv', tokenize=False)
# preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), 'generation')
# lr = LogisticModel(preprocessor_df_gen).fit()
# print(lr)

## SVM
df_gen = Reader('datasets/birth_year.csv', tokenize=False)#.dataset()
X = FastTextVectorizer(df_gen['post'].values).transform()
svm = SVMModel(df_gen, X, df_gen, 'generation').fit()
print(svm)
