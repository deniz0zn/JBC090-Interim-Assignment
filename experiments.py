from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from models import LogisticModel, FastTextVectorizer, SVMModel, DataPreprocessor
from imblearn.pipeline import Pipeline
from config import PL_PATH, BY_PATH, __DEBUG__
from reader import Reader
from evaluation import Evaluator
from fine_tune import fine_tune_log_reg, fine_tune_svm
from models import LogisticModel, FastTextVectorizer, SVMModel
# from explainable_ai import LimeEvaluator


class RunModels:
    """
    Run all models for a dataset.
    """
    def __init__(self, preprocessor: DataPreprocessor, tfidf: TfidfVectorizer, file_path):
        self.preprocessor = preprocessor
        self.tfidf = tfidf
        self.file_path = file_path
        self.df = Reader(self.file_path, tokenize=False)
        # self.X_train_vec, _ = self.preprocessor.vectorize_train()  # for log-reg
        self.X = FastTextVectorizer(self.df.dataset()['post'].values).transform()  # for SVM

    def run_default_logistic_regression(self):
        return LogisticModel(DataPreprocessor(self.df, self.tfidf, self.preprocessor.mode)).fit()

    def run_fine_tuned_logistic_regression(self):
        return LogisticModel(DataPreprocessor(self.df, self.tfidf, self.preprocessor.mode), fine_tuned = True).fit()

    def run_default_svm(self):
        return SVMModel(reader=self.df, X=self.X, target=self.preprocessor.mode).fit()

    def run_fine_tuned_svm(self):
        return SVMModel(self.df, self.X, self.preprocessor.mode, fine_tuned=True).fit()

    def add_predicted_political_leaning(self, model):  # svm fine tuned used
        if self.preprocessor.mode == "generation":
            self.df.dataset()['predicted_political_leaning'] = model.predict(self.X)
            self.df.dataset()['predicted_political_leaning'] = self.df.dataset()['predicted_political_leaning'].map({0: 'left',1: 'center',2: 'right'})
        return self.df



df_gen = Reader('datasets/birth_year.csv', tokenize=False)
preprocessor_df_gen = DataPreprocessor(df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), 'generation')
# lr = RunModels(preprocessor_df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), 'datasets/birth_year.csv')
# lr.run_default_logistic_regression()
# lr.run_fine_tuned_logistic_regression()
svm = RunModels(preprocessor_df_gen, TfidfVectorizer(use_idf=True, max_df=0.95), 'datasets/birth_year.csv')
svm.run_fine_tuned_svm()
### update: both svm models worked!!


class Experiment_setup:
    def __init__(self, by_path: str, pl_path: str):
        self.pl_reader = Reader(pl_path, tokenize=True)
        self.by_reader = Reader(by_path, tokenize=True)
        self.vectorizer = None
        self.preprocessed_data ={}

    def preprocess_data(self, reader: Reader, mode: str, pipeline):
        """
        Preprocess data for the specified reader and mode.
        Caches results for reuse.
        """
        if mode not in self.preprocessed_data:
            print(f"Preprocessing data for {mode}...")
            preprocessor = DataPreprocessor(reader, vectorizer=pipeline.named_steps["vectorizer"], mode=mode)
            self.preprocessed_data[mode] = preprocessor.preprocess()
            self.vectorizer = preprocessor.vectorizer
        return self.preprocessed_data[mode]



    def run_pipeline(self, reader: Reader, mode: str, model:str, top_n = 5, fine_tune= False):
        pipeline = create_pipeline(model)
        X_train, X_test, y_train, y_test = self.preprocess_data(reader, mode, pipeline)

        if fine_tune:
            print(f"Fine-tuning {model} model for {mode}...")

            if model == "logistic":
                best_params = self.fine_tuning_logistic(reader, mode)
                pipeline.named_steps["classifier"].set_params(**best_params)

            elif model == "SVM":
                # Replace the SVM model with the fine-tuned model
                best_model = self.fine_tuning_svm(reader, mode)
                pipeline.named_steps["classifier"] = best_model

        print(f"Fitting {model} pipeline...")
        pipeline.fit(X_train, y_train)

        print(f"Evaluating {model} pipeline...")
        evaluator = Evaluator(pipeline)
        evaluator.evaluate(X_test, y_test, model_name=f"{model}_{mode}")
        evaluator.top_words(top_n=top_n)

    def save_predictions(self, reader: Reader, mode: str, model: str, fine_tuned=True):
        print(f"Saving predictions for {model} on {mode} dataset...")

        preprocessor = DataPreprocessor(reader, mode=mode)
        _, X_test, _, y_test = preprocessor.preprocess()

        if fine_tuned:
            if model == "logistic":
                best_params = self.fine_tuning_logistic(reader, mode)
                pipeline = create_pipeline(model)
                pipeline.named_steps["classifier"].set_params(**best_params)
            elif model == "SVM":
                pipeline = create_pipeline(model)
                pipeline.named_steps["classifier"] = self.fine_tuning_svm(reader, mode)
        else:
            pipeline = create_pipeline(model)

        predictions = pipeline.predict(X_test)
        reader.dataset["predicted_political_leaning"] = predictions
        print(f"Predictions saved to dataset.")

    def log_metrics(self, evaluator: Evaluator, model_name: str):
        metrics = evaluator.results.get(model_name, {})
        print(f"Metrics for {model_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


class Experiments:
    """
    Manages the three phases of experimentation with Logistic Regression and SVM pipelines.
    """
    def __init__(self, experiment_setup):
        self.experiment_setup = experiment_setup

    def phase_1(self):
        """
        Logistic Regression Baseline
        """
        print("--- Logistic Regression Baseline ---")


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="logistic", fine_tune=False
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.pl_reader, mode="political_leaning", model="logistic", fine_tune=False
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="logistic", fine_tune=True
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.pl_reader, mode="political_leaning", model="logistic", fine_tune=True
        )


        self.experiment_setup.save_predictions(
    reader=self.experiment_setup.by_reader, mode="political_leaning", model="logistic", fine_tuned=True
)


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="logistic", fine_tune=True
        )


        evaluator = Evaluator(None)
        evaluator.explain_predictions(dataset=self.experiment_setup.by_reader.dataset, text_column="post")

    def phase_2(self):
        """
        FastText + SVM
        """
        print("--- FastText + SVM ---")


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="SVM", fine_tune=False
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.pl_reader, mode="political_leaning", model="SVM", fine_tune=False
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="SVM", fine_tune=True
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.pl_reader, mode="political_leaning", model="SVM", fine_tune=True
        )


        self.experiment_setup.save_predictions(
            reader=self.experiment_setup.by_reader, mode="political_leaning", model="SVM", fine_tuned=True
        )


        self.experiment_setup.run_pipeline(
            reader=self.experiment_setup.by_reader, mode="birth_year", model="SVM", fine_tune=True
        )


        evaluator = Evaluator(None)
        evaluator.explain_predictions(dataset=self.experiment_setup.by_reader.dataset, text_column="post")

    def phase_3(self):
        """
        Metrics
        """
        print("--- Metrics ---")

        # 35. Print ROC and F1 scores
        evaluator = Evaluator(None)
        evaluator.log_metrics(None, "logistic")  # Replace "logistic" with specific model name as needed

        # 36. Print top 5 words for the models
        evaluator.top_words(top_n=5)



setup = Experiment_setup(BY_PATH, PL_PATH)
experiments = Experiments(setup)
experiments.phase_1()
experiments.phase_2()
experiments.phase_3()







