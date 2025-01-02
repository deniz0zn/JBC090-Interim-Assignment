import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reader import Reader, DataLoader
from models import LogisticModel, DistilBERTModel
from fine_tune import fine_tune_logistic, fine_tune_transformer
from evaluation import Evaluation

class ExperimentPipeline:
    def __init__(self):
        """
        Initializes the pipelines for running experiments.
        """
        self.logistic_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(use_idf=True, max_df=0.95)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

    def run_logistic_pipeline(self, train_data, test_data, target_column, evaluator, experiment_name):
        """
        Runs logistic regression pipeline.

        Parameters:
            train_data: Training dataset.
            test_data: Testing dataset.
            target_column: Name of the target column.
            evaluator: Instance of the Evaluation class.
            experiment_name: Name of the experiment.
        """
        X_train, y_train = train_data["post"], train_data[target_column]
        X_test, y_test = test_data["post"], test_data[target_column]

        self.logistic_pipeline.fit(X_train, y_train)
        predictions = self.logistic_pipeline.predict(X_test)

        evaluator.evaluate(
            self.logistic_pipeline.named_steps['classifier'],
            self.logistic_pipeline.named_steps['vectorizer'].transform(X_test),
            y_test, experiment_name
        )
        evaluator.top_words(
            self.logistic_pipeline.named_steps['classifier'],
            self.logistic_pipeline.named_steps['vectorizer']
        )

class Experiment:
    def __init__(self, birth_year_path, political_leaning_path):
        """
        Initialize the experiment with dataset paths.

        Parameters:
            birth_year_path: Path to the birth_year dataset.
            political_leaning_path: Path to the political_leaning dataset.
        """
        self.birth_year_path = birth_year_path
        self.political_leaning_path = political_leaning_path
        self.evaluator = Evaluation()
        self.pipeline = ExperimentPipeline()

    def logistic_experiments(self, birth_year_data, political_leaning_data):
        """
        Run all logistic regression experiments.
        """
        print("\nRunning Logistic Regression Pipeline Experiments...")
        self.pipeline.run_logistic_pipeline(
            birth_year_data.train,
            birth_year_data.test,
            "birth_year",
            self.evaluator,
            "Logistic on Birth Year"
        )
        self.pipeline.run_logistic_pipeline(
            political_leaning_data.train,
            political_leaning_data.test,
            "political_leaning",
            self.evaluator,
            "Logistic on Political Leaning"
        )

    def distilbert_experiments(self, birth_year_data, political_leaning_data):
        """
        Run all DistilBERT experiments.
        """
        print("\nRunning DistilBERT Experiments...")
        distil_model = DistilBERTModel(birth_year_data.train)
        model_instance, tokenizer = distil_model.train()
        self.evaluator.evaluate(
            model_instance,
            tokenizer.encode_plus(
                birth_year_data.test["post"].tolist(),
                truncation=True, padding=True, return_tensors="pt"
            )["input_ids"],
            birth_year_data.test["birth_year"],
            "DistilBERT on Birth Year"
        )

        distil_model = DistilBERTModel(political_leaning_data.train)
        model_instance, tokenizer = distil_model.train()
        self.evaluator.evaluate(
            model_instance,
            tokenizer.encode_plus(
                political_leaning_data.test["post"].tolist(),
                truncation=True, padding=True, return_tensors="pt"
            )["input_ids"],
            political_leaning_data.test["political_leaning"],
            "DistilBERT on Political Leaning"
        )

    def fine_tune_logistic(self, birth_year_data):
        """
        Fine-tune logistic regression for birth_year.
        """
        print("\nRunning Fine-Tuned Logistic Regression...")
        logistic_model = LogisticModel(birth_year_data.train)
        model_instance, vectorizer = logistic_model.train()
        fine_tuned_model, _ = fine_tune_logistic(
            model_instance,
            vectorizer.transform(birth_year_data.train["post"]),
            birth_year_data.train["birth_year"]
        )
        self.evaluator.evaluate(
            fine_tuned_model,
            vectorizer.transform(birth_year_data.test["post"]),
            birth_year_data.test["birth_year"],
            "Fine-Tuned Logistic on Birth Year"
        )

    def run(self):
        """
        Run all experiments as defined in the task.
        """
        # Load datasets using DataLoader
        birth_year_loader = DataLoader(self.birth_year_path, "birth_year")
        political_leaning_loader = DataLoader(self.political_leaning_path, "political_leaning")

        birth_year_data = birth_year_loader.load()
        political_leaning_data = political_leaning_loader.load()

        # Run Logistic Regression Experiments
        self.logistic_experiments(birth_year_data, political_leaning_data)

        # Run Fine-Tuned Logistic Regression
        self.fine_tune_logistic(birth_year_data)

        # Run DistilBERT Experiments
        self.distilbert_experiments(birth_year_data, political_leaning_data)

if __name__ == "__main__":

    experiment = Experiment("datasets/birth_year.csv", "datasets/political_leaning.csv")
    experiment.run()
