import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from models import LogisticModel, FastTextVectorizer, SVMModel, DataPreprocessor
from config import PL_PATH, BY_PATH, __DEBUG__
from reader import Reader, Dataset
from fine_tune import X_with_pred_pol_lean, run_fine_tuned_log_with_pol
from explainable_ai import LimeEvaluator


class RunModels:
    """
    Class to run all models for a dataset.
    Print unique features if dataset contains `generation` target.
    :attributes: preprocessor, file_path
    """
    def __init__(self, preprocessor: DataPreprocessor, file_path: str) -> None:
        """
        Initializes a RunModels object.
        :param preprocessor: DataPreprocessor to vectorize features for logistic regression
        :param file_path: path to dataset
        """
        self.preprocessor = preprocessor
        self.vectorizer = self.preprocessor.vectorizer
        self.file_path = file_path
        self.df = self.preprocessor.reader
        self.X = FastTextVectorizer(self.df.dataset()['post'].values).transform()  # for SVM
        print(f"\nInitialized RunModels object for '{self.preprocessor.mode}'.\n")

    def run_default_logistic_regression(self) -> LogisticRegression:
        """
        Run logistic regression using default parameters.
        :return: logistic regression model with default parameters.
        """
        print(f"\nRunning default logistic regression for '{self.preprocessor.mode}'...\n")
        return LogisticModel(self.preprocessor).fit()

    def run_fine_tuned_logistic_regression(self, pred_pol=False) -> LogisticRegression:
        """
        Run logistic regression using fine-tuned parameters.
        :return: logistic regression model with fine-tuned parameters.
        """
        print(f"\nRunning fine-tuned logistic regression for '{self.preprocessor.mode}'...\n")
        return LogisticModel(self.preprocessor, fine_tuned=True, pred_pol=pred_pol).fit()

    def run_default_svm(self) -> SVC:
        """
        Run SVM using default parameters.
        :return: SVM model with default parameters.
        """
        print(f"\nRunning default SVM for '{self.preprocessor.mode}'...\n")
        return SVMModel(reader=self.df, X=self.X, target=self.preprocessor.mode).fit()

    def run_fine_tuned_svm(self) -> SVC:
        """
        Run SVM using fine-tuned parameters.
        :return: SVM model with fine-tuned parameters.
        """
        print(f"\nRunning fine-tuned SVM for '{self.preprocessor.mode}'...\n")
        return SVMModel(reader=self.df, X=self.X, target=self.preprocessor.mode, fine_tuned=True).fit()

    @staticmethod
    def predict_political_leaning(df: pd.DataFrame, model: LogisticRegression, vectorizer: TfidfVectorizer) -> pd.DataFrame:
        """
        Predict political leaning using fine-tuned model.
        :param df: dataframe to predict political leaning on
        :param model: fine-tuned model to use
        :param vectorizer: vectorizer to use
        :return: dataframe with predicted political leaning.
        """
        print(f"Adding the predicted political leaning column to the 'generation' dataset.")
        df["predicted_political_leaning"] = model.predict(vectorizer.transform(df["post"]))
        df["predicted_political_leaning"] = df["predicted_political_leaning"].map({0: "left", 1: "center", 2: "right"})
        return df

    @staticmethod
    def run_explainability(df: pd.DataFrame, model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
        """
        Add predicted political leaning column to dataframe.
        Use Lime to print unique features for each generation and its prediction.
        :param df: dataframe to perform explainability on.
        :param model: model that predicts and explains political leaning
        :param vectorizer: vectorizer to use
        """
        print("Running explainability with Lime...")
        explainer = LimeEvaluator(df, model, vectorizer)
        explainer.unique_all_generations(explainer.explain())


def run_experiments(path: str, mode: str):
    """
    Run all default and fine-tuned models.
    :param path: path to file
    :param mode: target
    """
    print(f"Processing dataset at {path} for target mode: {mode}...")
    dataset = Dataset(path)
    print(dataset)  # Print dataset summary
    reader = Reader(path, tokenize=True)
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
    preprocessor = DataPreprocessor(reader, vectorizer, mode)
    run_models = RunModels(preprocessor, path)

    print("--- Running Logistic Regression Models ---")
    run_models.run_default_logistic_regression()
    logistic_model = run_models.run_fine_tuned_logistic_regression()

    print("--- Running SVM Models ---")
    run_models.run_default_svm()
    run_models.run_fine_tuned_svm()
    print(reader.dataset())

    if mode == "generation":
        return reader.dataset(), run_models
    else:
        return logistic_model, vectorizer


def run_with_pol_lean(df, model, vectorizer, run_models):
    df_with_predictions = run_models.predict_political_leaning(df=df, model=model, vectorizer=vectorizer)
    X_with_pred_pol = X_with_pred_pol_lean(df=df, tfidf=vectorizer, model=model)
    pred_model = run_fine_tuned_log_with_pol(X=X_with_pred_pol, df=df_with_predictions)
    run_models.run_explainability(df=df_with_predictions, model=pred_model, vectorizer=vectorizer)


if __name__ == "__main__":
    t0 = time.time()

    print("--- BIRTH YEAR DATASET ---")
    df_generation, run_class = run_experiments(BY_PATH, "generation")

    print("--- POLITICAL LEANING DATASET ---")
    political_leaning_model, political_leaning_vectorizer = run_experiments(PL_PATH, "political_leaning")

    print("--- Predicting Political Leaning for Generation Dataset ---")
    run_with_pol_lean(df_generation, political_leaning_model, political_leaning_vectorizer, run_class)

    print(f"Total Time Taken: {round(((time.time() - t0) / 60), 2)} minutes")
