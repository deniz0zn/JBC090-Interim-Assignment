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
import joblib


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
        self.df = self.preprocessor.reader # Reader(self.file_path, tokenize=True)
        # self.X = FastTextVectorizer(self.df.dataset()['post'].values).transform()  # for SVM

    def run_default_logistic_regression(self) -> LogisticRegression:
        """
        Run logistic regression using default parameters.
        :return: logistic regression model with default parameters.
        """
        print("Running default logistic regression...")
        # return LogisticModel(DataPreprocessor(self.df, self.tfidf, self.preprocessor.mode)).fit()
        return LogisticModel(DataPreprocessor(self.df, self.vectorizer, self.preprocessor.mode)).fit()

    def run_fine_tuned_logistic_regression(self, pred_pol=False) -> LogisticRegression:
        """
        Run logistic regression using fine-tuned parameters.
        :return: logistic regression model with fine-tuned parameters.
        """
        return LogisticModel(DataPreprocessor(self.df, self.vectorizer, self.preprocessor.mode), fine_tuned=True, pred_pol=pred_pol).fit()
        print("Running fine-tuned logistic regression...")
        # return LogisticModel(DataPreprocessor(self.df, self.tfidf, self.preprocessor.mode), fine_tuned=True, pred_pol=pred_pol).fit()

    def run_default_svm(self) -> SVC:
        """
        Run SVM using default parameters.
        :return: SVM model with default parameters.
        """
        print("Running default SVM...")
        return SVMModel(reader=self.df, X=self.X, target=self.preprocessor.mode).fit()

    def run_fine_tuned_svm(self) -> SVC:
        """
        Run SVM using fine-tuned parameters.
        :return: SVM model with fine-tuned parameters.
        """
        print("Running fine-tuned SVM...")
        return SVMModel(reader=self.df, X=self.X, target=self.preprocessor.mode, fine_tuned=True).fit()

    def predict_political_leaning(self, model: LogisticRegression) -> pd.DataFrame:
        """
        Predict political leaning using fine-tuned model.
        :param model: fine-tuned model to use
        :return: dataframe with predicted political leaning.
        """
        if self.preprocessor.mode == "generation":
            self.df.dataset()["predicted_political_leaning"] = model.predict(self.vectorizer.transform(self.df.dataset()["generation"]))
            self.df.dataset()["predicted_political_leaning"] = self.df.dataset()["predicted_political_leaning"].map({0: "left", 1: "center", 2: "right"})
        return self.df.dataset()

    def run_explainability(self, df: pd.DataFrame, model: LogisticRegression) -> None:
        """
        If target of dataset is `generation`, add predicted political leaning column to dataframe.
        Use Lime to print unique features for each generation and its prediction.
        :param model: model that predicts and explains political leaning
        :return:
        """
        explainer = LimeEvaluator(df, model, self.vectorizer)
        print("Running explainability with Lime...")
        # explainer = LimeEvaluator(df, model, self.tfidf)
        explainer.unique_all_generations(explainer.explain())


t0 = time.time()  # see time taken
# -- BIRTH YEAR--
df = Dataset(BY_PATH)
# print(df)  # print information about birth_year dataset
# df_gen = Reader(BY_PATH, tokenize=True)
df_gen = Reader('datasets/birth_year_tokenized_cleaned.csv', False)
print('df gen')
preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), "generation")
print('preprocess')
run_models = RunModels(preprocessor_df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), 'datasets/birth_year_tokenized_cleaned.csv')
print('lrmodel')
# run_models.run_default_logistic_regression()  # logistic regression with default parameters

lr_model = run_models.run_fine_tuned_logistic_regression()  # logistic regression with fine-tuned parameters
# joblib.dump(lr_model, 'models/lr_model.pkl')

# run_models.run_default_svm()  # svm with default parameters
# run_models.run_fine_tuned_svm()  # svm with fine-tuned parameters

# lr_model = joblib.load('models/lr_model.pkl')
df_pred_pol = run_models.predict_political_leaning(lr_model)  # add `predicted_political_leaning` column
print(df_pred_pol)
# df_pred_pol = pd.read_parquet('datasets/birth_year_with_political_leaning.parquet')
# print(df_pred_pol.head())
X_with_pred_pol = X_with_pred_pol_lean(df=df_pred_pol, tfidf=run_models.tfidf, model=lr_model)  # X has 2 features now
pred_model = run_fine_tuned_log_with_pol(X=X_with_pred_pol, df=df_pred_pol)  # logistic regression with fine-tuned parameters and new X
run_models.run_explainability(df=df_pred_pol, model=pred_model)  # run AI explainability on added column


# # -- POLITICAL LEANING --
# df = Dataset(PL_PATH)
# print(df)  # print information about political leaning dataset
# df_gen = Reader(PL_PATH, tokenize=True)
# preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), "political_leaning")
# run_models = RunModels(preprocessor_df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), PL_PATH)
# run_models.run_default_logistic_regression()  # logistic regression with default parameters
# run_models.run_fine_tuned_logistic_regression()  # logistic regression with fine-tuned parameters
# run_models.run_default_svm()  # svm with default parameters
# run_models.run_fine_tuned_svm()  # svm with fine-tuned parameters
print(f"Time taken: {round(((time.time() - t0) / 60), 2)} minutes")
# t0 = time.time()  # see time taken
# # -- BIRTH YEAR--
# df = Dataset(BY_PATH)
# print(df)  # print information about birth_year dataset
# df_gen = Reader(BY_PATH, tokenize=True)
# preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), "generation")
# run_models = RunModels(preprocessor_df_gen, BY_PATH)
# run_models.run_default_logistic_regression()  # logistic regression with default parameters
# lr_model = run_models.run_fine_tuned_logistic_regression()  # logistic regression with fine-tuned parameters
# run_models.run_default_svm()  # svm with default parameters
# run_models.run_fine_tuned_svm()  # svm with fine-tuned parameters
# df_pred_pol = run_models.predict_political_leaning(lr_model)  # add `predicted_political_leaning` column
# X_with_pred_pol = X_with_pred_pol_lean(df=df_pred_pol, tfidf=run_models.vectorizer, model=lr_model)  # X has 2 features now
# pred_model = run_fine_tuned_log_with_pol(X=X_with_pred_pol, df=df_pred_pol)  # logistic regression with fine-tuned parameters and new X
# run_models.run_explainability(df=df_pred_pol, model=pred_model)  # run AI explainability on added column
#
# # -- POLITICAL LEANING --
# df = Dataset(PL_PATH)
# print(df)  # print information about political leaning dataset
# df_gen = Reader(PL_PATH, tokenize=True)
# preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), "political_leaning")
# run_models = RunModels(preprocessor_df_gen, PL_PATH)
# run_models.run_default_logistic_regression()  # logistic regression with default parameters
# run_models.run_fine_tuned_logistic_regression()  # logistic regression with fine-tuned parameters
# run_models.run_default_svm()  # svm with default parameters
# run_models.run_fine_tuned_svm()  # svm with fine-tuned parameters
# print(f"Time taken: {round(((time.time() - t0) / 60), 2)} minutes")



def run_experiments(path: str, mode):
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

    if mode == "generation":
        print("--- Predicting Political Leaning for Generation Dataset ---")
        df_with_predictions = run_models.predict_political_leaning(logistic_model)
        X_with_pred_pol = X_with_pred_pol_lean(df=df_with_predictions, tfidf=vectorizer, model=logistic_model)
        pred_model = run_fine_tuned_log_with_pol(X=X_with_pred_pol, df=df_with_predictions)
        run_models.run_explainability(df=df_with_predictions, model=pred_model)


if __name__ == "__main__":
    t0 = time.time()

    print("--- BIRTH YEAR DATASET ---")
    run_experiments(BY_PATH, "generation")

    print("--- POLITICAL LEANING DATASET ---")
    run_experiments(PL_PATH, "political_leaning")

    print(f"Total Time Taken: {round(((time.time() - t0) / 60), 2)} minutes")
