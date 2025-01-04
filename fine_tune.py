from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from config import param_grid, __DEBUG__, CV, svm_C_values, svm_gamma_values, __RANDOM_STATE__
import os


def fine_tune_svm(model_path, X_train, y_train, X_test, y_test, DEBUG=__DEBUG__):
    """
    Fine-tune SVM model using C values and gamma values.
    Parameters:
        model_path: path to saved fine-tuned model
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
        DEBUG: debug flag
    """
    if DEBUG and os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}...")
        best_model = load(model_path)
        return best_model

    best_score = 0
    best_model = None

    for C in svm_C_values:
        for gamma in svm_gamma_values:
            print(f"Trying C={C}, gamma={gamma}")
            svm = SVC(kernel='rbf',
                      C=C,
                      gamma=gamma,
                      random_state=__RANDOM_STATE__)
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            print(f"Score for C={C}, gamma={gamma}: {score}")

            if score > best_score:
                best_model = svm
                best_score = score

    return best_model


def fine_tune_log_reg(model_path, vectorizer, vectorizer_path, X_train, y_train, param_grid=param_grid,
                      cv=CV, DEBUG=__DEBUG__):
    """
    Fine-tune logistic regression model using GridSearchCV.
    Save fine-tuned model and vectorizer.
    Parameters:
        model_path: path to saved fine-tuned model
        vectorizer: vectorizer
        vectorizer_path: path to saved vectorizer
        X_train: training features
        y_train: training labels
        param_grid: parameter grid
        cv: cross validation split
        DEBUG: debug flag
    """
    if DEBUG and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print(f"Loading fine-tuned model from {model_path} and vectorizer from {vectorizer_path}...")
        best_model = load(model_path)
        vectorizer = load(vectorizer_path)
        return best_model, vectorizer, None

    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=500),
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=10
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    dump(grid_search.best_estimator_, model_path)
    dump(vectorizer, vectorizer_path)

    print(f"Fine-tuned model saved to {model_path}.")
    print(f"Vectorizer saved to {vectorizer_path}.")

    return grid_search.best_estimator_, vectorizer, None
