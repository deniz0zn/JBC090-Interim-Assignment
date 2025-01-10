from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, log_loss, \
    matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models.fasttext import FastText
from config import __RANDOM_STATE__, test_size, __DEBUG__, param_grid
from reader import Reader
from config_fine_tune import *
from fine_tune import *


class DataPreprocessor:
    """
    Preprocess features for logistic regression using a vectorizer.
    :attributes: reader, vectorizer, mode, test_size, random_state
    """
    def __init__(self, reader: Reader.dataset, vectorizer, mode, test_size=test_size, random_state=__RANDOM_STATE__) -> None:
        """
        Initializes a DataPreprocessor class.
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
        self.X_train, self.X_test, self.y_train, self.y_test = self.reader.split_data(self.test_size, self.random_state)

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
        self.y_train, self.y_test = self.mapping(self.y_train, self.y_test)

        print("Fitting the vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        print("Vectorizer fitted.\n")

        return X_train_vec, X_test_vec, self.y_train, self.y_test

    def vectorize_train(self):
        """
        Vectorize training data.
        """
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)

        return X_train_vec, X_test_vec


class LogisticModel:
    """
    Run logistic regression model with default parameters and balanced class weights.
    :attributes: preprocessor, fine_tuned, pred_pol, debug
    """
    def __init__(self, preprocessor: DataPreprocessor, fine_tuned=False, pred_pol=False, debug=__DEBUG__) -> None:
        """
        Initializes a LogisticModel class.
        :param preprocessor: DataPreprocessor to preprocess the data
        :param fine_tuned: run fine-tuned model or not
        :param pred_pol: run with predicted political leaning or not
        :param debug: debug mode
        """
        self.preprocessor = preprocessor
        self.vectorizer = preprocessor.vectorizer
        self.fine_tuned = fine_tuned
        self.pred_pol = pred_pol
        self.debug = debug

        if self.fine_tuned:
            if self.pred_pol:
                self.X_train = X_with_pred_pol_lean(self.preprocessor.reader.dataset(), self.vectorizer)
                print('X_train for predicted political leaning used.')
            else:
                self.X_train, __ = self.preprocessor.vectorize_train()
            parameters = fine_tune_log_reg(X_train=self.X_train, y_train=self.preprocessor.y_train,
                                           DEBUG=self.debug, mode=self.preprocessor.mode)
            solver, penalty, C = parameters["solver"], parameters["penalty"], parameters["C"]

            self.model = LogisticRegression(class_weight="balanced", penalty=penalty, C=C, solver=solver,
                                            max_iter=1000, random_state=self.preprocessor.random_state
                                            )
        else:
            self.model = LogisticRegression(class_weight="balanced",
                                            max_iter=1000, random_state=self.preprocessor.random_state
                                            )

    def fit(self) -> LogisticRegression:
        """
        Fit logistic regression model with default parameters and balanced class weights and print metrics.
        :return: fitted logistic regression model.
        """
        X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

        print(f"Fitting the model with parameters: {self.model.get_params()}\n")
        self.model.fit(X_train_vec, y_train)
        print(f"Model fitted. Metrics:\n{Metrics(X_test_vec, y_test, self.model)}")

        return self.model


class FastTextVectorizer:
    """
    Preprocess features for SVM using FastText vectorizer.
    :attributes: texts, vector_size, window, min_count
    """
    def __init__(self, texts, vector_size=100, window=5, min_count=1) -> None:
        """
        Initializes a FastTextVectorizer class.
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

    def create_model(self) -> FastText:
        """
        Create FastText embeddings model.
        :return: FastText word embeddings model.
        """
        processed_texts = [text.split() for text in self.texts]
        self.model = FastText(sentences=processed_texts,
                              vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              workers=4
                              )
        return self.model

    def transform(self) -> np.array:
        """
        Fit FastText embeddings model with default parameters.
        :return: vectorized features.
        """
        if not self.model:
            self.model = self.create_model()
        return np.array([self.get_text_embedding(text) for text in self.texts])

    def get_text_embedding(self, text: str) -> np.array:
        """
        Return text embeddings for feature X.
        :param text: text to vectorize
        :return: embedding vector.
        """
        words = text.split()
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)


class SVMModel:
    """
    Run logistic regression model with default parameters and balanced class weights.
    :attributes: reader, X, target, fine_tuned, test_size, random_state, debug
    """
    def __init__(self, reader: Reader, X, target: str, fine_tuned=False, test_size=test_size, random_state=__RANDOM_STATE__, debug=__DEBUG__) -> None:
        """
        Initializes an SVMModel class.
        :param reader: read  and clean data
        :param X: vectorized features
        :param target: target to predict
        :param fine_tuned: run fine-tuned model or not
        :param test_size: size of test data
        :param random_state: seed for reproducibility
        :param debug: debug mode
        """
        self.reader = reader.dataset()
        self.X = X
        self.target = target
        self.fine_tuned = fine_tuned
        self.test_size = test_size
        self.random_state = random_state
        self.debug = debug

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.reader[self.target],
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state
                                                                                )
        if not self.fine_tuned:  # use default parameters
            self.model = SVC(probability=True,
                             class_weight="balanced",
                             random_state=self.random_state
                             )
        else:
            parameters = fine_tune_svm(X_train=self.X_train,
                                       y_train=self.reader[self.target],
                                       X_test=self.X_test,
                                       y_test=self.reader[self.target],
                                       DEBUG=self.debug)
            C, gamma = parameters["C"], parameters["gamma"]
            self.model = SVC(probability=True,
                             class_weight="balanced",
                             C=C,
                             gamma=gamma,
                             random_state=self.random_state
                             )

    def fit(self) -> SVC:
        """
        Fit SVM model and print metrics.
        :return: fitted SVM model.
        """
        print(f"Fitting the model with parameters: {self.model.get_params()}\n")
        self.model.fit(self.X_train, self.y_train)
        print(f"SVM model fitted. Metrics:\n{Metrics(self.X_test, self.y_test, self.model)}")

        return self.model
