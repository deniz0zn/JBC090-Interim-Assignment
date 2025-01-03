from sklearn.metrics import classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
from tqdm import tqdm
import shap

class Evaluator:
    """
    Handles model evaluation, including metrics calculation, ROC curve plotting,
    feature importance reporting for interpretability, and prediction explanation.
    """

    def __init__(self,model,vectorizer):
        self.results = {}
        self.model = model
        self.vectorizer = vectorizer

    def evaluate(self,model, X_test, y_test, model_name="Model"):
        """
        Evaluate the model on the test set.

        Parameters:
            model: Trained model.
            X_test: Features of the test set.
            y_test: True labels of the test set.
            model_name: Name of the model for result tracking.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

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

    def top_words(self, model, vectorizer, top_n=5):
        """
        Display the top words/features for interpretability.

        Parameters:
            model: Trained model.
            vectorizer: Fitted vectorizer for feature extraction.
            top_n: Number of top features to display.
        """
        if not hasattr(model, "coef_"):
            print("Feature importance is unavailable for this model.")
            return

        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        top_positive = sorted(zip(coefs, feature_names), reverse=True)[:top_n]
        top_negative = sorted(zip(coefs, feature_names))[:top_n]

        print("Top Positive Features:")
        for coef, feature in top_positive:
            print(f"{feature}: {coef:.4f}")

        print("\nTop Negative Features:")
        for coef, feature in top_negative:
            print(f"{feature}: {coef:.4f}")

    def predict_and_update(self, dataset, text_column="post", key_column="author_id"):
        X = self.vectorizer.transform(tqdm(dataset[text_column], desc="Vectorizing Dataset"))
        predictions = self.model.predict(X)
        predictions_df = pd.DataFrame({
            key_column: dataset[key_column],
            "predicted_label": predictions
        })
        updated_dataset = pd.merge(dataset, predictions_df, on=key_column, how="left")
        return updated_dataset

    def explain_predictions(self, model, vectorizer, dataset, text_column="post"):
        """
        Explains the model's predictions using SHAP.
        """
        X = vectorizer.transform(tqdm(dataset[text_column], desc="Preparing SHAP Data"))
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X, nsamples=100)
        shap.summary_plot(shap_values, X, feature_names=vectorizer.get_feature_names_out())
