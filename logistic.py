import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from joblib import dump, load
import shap
from tqdm import tqdm

from config import param_grid, __RANDOM_STATE__



class DataPreprocessor:
    def __init__(self, dataset, text_column="post", target_column="target", test_size=0.3, random_state=__RANDOM_STATE__, mode="birth"): # birth_year or political_leaning
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.mode = mode
        self.vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)

    def preprocess(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset[self.text_column],
            self.dataset[self.target_column],
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        if self.mode == "political_leaning":
            y_train = y_train.map({'left': 0, 'center': 1, 'right': 2})
            y_test = y_test.map({'left': 0, 'center': 1, 'right': 2})

        self.vectorizer.fit(X_train)
        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        return X_train_vec, X_test_vec, y_train, y_test



class ModelTrainer:
    def __init__(self, preprocessor, debug=False):
        self.preprocessor = preprocessor
        self.model = LogisticRegression()
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

    def fine_tune(self, save_path_model, save_path_vectorizer):
        if self.debug and os.path.exists(save_path_model) and os.path.exists(save_path_vectorizer):
            print("DEBUG mode enabled. Loading existing fine-tuned model.")
            self.model = load(save_path_model)
            self.preprocessor.vectorizer = load(save_path_vectorizer)
            return

        X_train_vec, X_test_vec, y_train, y_test = self.preprocessor.preprocess()

        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=500),
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            verbose=10
        )
        grid_search.fit(X_train_vec, y_train)

        self.model = grid_search.best_estimator_
        self.parameters_after_tuning = grid_search.best_params_

        print("Best Parameters:", grid_search.best_params_)
        print("Parameters after tuning:", self.parameters_after_tuning)

        dump(self.model, save_path_model)
        dump(self.preprocessor.vectorizer, save_path_vectorizer)
        print(f"Fine-tuned model and vectorizer saved to {save_path_model} and {save_path_vectorizer}.")


class ModelEvaluator:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict_and_update(self, dataset, text_column="post", key_column="author_id"):
        new_X_vec = self.vectorizer.transform(dataset[text_column])

        predictions = []
        for i in tqdm(range(new_X_vec.shape[0]), desc="Making Predictions"):
            predictions.append(self.model.predict(new_X_vec[i]))

        predictions = [pred[0] for pred in predictions]

        predictions_df = pd.DataFrame({key_column: dataset[key_column], "predictions": predictions})
        updated_dataset = dataset.merge(predictions_df, on=key_column, how='left')

        return updated_dataset

    def explain_predictions(self, dataset):
        explainer = shap.LinearExplainer(
            self.model, self.vectorizer.transform(dataset[self.preprocessor.text_column]),
            feature_perturbation='interventional')
        shap_values = explainer.shap_values(self.vectorizer.transform(dataset[self.preprocessor.text_column]))

        shap.summary_plot(
            shap_values,
            features=self.vectorizer.transform(dataset[self.preprocessor.text_column]),
            feature_names=self.vectorizer.get_feature_names_out(),
        )



