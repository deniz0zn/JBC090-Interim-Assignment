from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
import pandas as pd
from tqdm import tqdm
import shap

class Evaluator:
    """
    Handles model evaluation, including metrics calculation, ROC curve plotting,
    feature importance reporting for interpretability, and prediction explanation.
    """

    def __init__(self, pipeline):
        self.results = {}
        self.pipeline = pipeline
        self.vectorizer = self._extract_component("vectorizer")
        self.classifier = self._extract_component("classifier")

    def _extract_component(self, component_name):
        """
        Extracts a component from the pipeline if it exists.
        """
        if isinstance(self.pipeline, Pipeline) and component_name in self.pipeline.named_steps:
            return self.pipeline.named_steps[component_name]
        return None

    def evaluate(self, X_test, y_test, model_name="Model"):
        """
        Evaluate the pipeline on the test set.

        Parameters:
            X_test: Features of the test set.
            y_test: True labels of the test set.
            model_name: Name of the model for result tracking.
        """
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test) if hasattr(self.pipeline, "predict_proba") else None

        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = None
        if y_proba is not None and len(set(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            self.plot_roc(y_test, y_proba, model_name)

        self.results[model_name] = {
            "classification_report": report,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        print(f"Evaluation for {model_name}:")
        print(pd.DataFrame(report).transpose())
        print(f"F1 Score: {f1}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc}")

    def plot_roc(self, y_test, y_proba, model_name):
        """
        Plot ROC curve.

        Parameters:
            y_test: True labels of the test set.
            y_proba: Predicted probabilities.
            model_name: Name of the model for the plot title.
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_proba[:, 1]):.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    def top_words(self, top_n=5):
        """
        Display the top words/features for interpretability.

        Parameters:
            top_n: Number of top features to display.
        """
        if not hasattr(self.classifier, "coef_") or self.vectorizer is None:
            print("Feature importance is unavailable for this pipeline.")
            return

        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.classifier.coef_[0]
        top_positive = sorted(zip(coefs, feature_names), reverse=True)[:top_n]
        top_negative = sorted(zip(coefs, feature_names))[:top_n]

        print("Top Positive Features:")
        for coef, feature in top_positive:
            print(f"{feature}: {coef:.4f}")

        print("\nTop Negative Features:")
        for coef, feature in top_negative:
            print(f"{feature}: {coef:.4f}")

    def predict_and_update(self, dataset, text_column="post", key_column="author_id"):
        """
        Predicts labels for a given dataset and updates it with predictions.

        Parameters:
            dataset: Input dataset for prediction.
            text_column: Column containing the text data.
            key_column: Unique identifier column for the dataset.

        Returns:
            Updated dataset with predictions.
        """
        X = self.vectorizer.transform(tqdm(dataset[text_column], desc="Vectorizing Dataset"))
        predictions = self.pipeline.predict(X)
        predictions_df = pd.DataFrame({
            key_column: dataset[key_column],
            "predicted_label": predictions
        })
        updated_dataset = pd.merge(dataset, predictions_df, on=key_column, how="left")
        return updated_dataset

    def explain_predictions(self, dataset, text_column="post"):
        """
        Explains the model's predictions using SHAP.

        Parameters:
            dataset: Input dataset for explanation.
            text_column: Column containing the text data.
        """
        X = self.vectorizer.transform(tqdm(dataset[text_column], desc="Preparing SHAP Data"))
        explainer = shap.KernelExplainer(self.pipeline.predict_proba, X)
        shap_values = explainer.shap_values(X, nsamples=100)
        shap.summary_plot(shap_values, X, feature_names=self.vectorizer.get_feature_names_out())
