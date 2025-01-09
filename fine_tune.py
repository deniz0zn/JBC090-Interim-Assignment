import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from config import param_grid, __DEBUG__, svm_C_values, svm_gamma_values, __RANDOM_STATE__
import os
from config_fine_tune import pol_logistic_parameters, gen_logistic_parameters, svm_parameters


def fine_tune_svm(X_train, y_train, X_test, y_test, DEBUG=__DEBUG__):
    """
    Fine-tune SVM model using C values and gamma values.
    Parameters:
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
        DEBUG: debug flag
    """
    if DEBUG:
        return svm_parameters

    best_score = 0
    best_model = None

    for C in svm_C_values:
        for gamma in svm_gamma_values:
            print(f"Trying C={C}, gamma={gamma}")
            svm = SVC(kernel="rbf",
                      class_weight="balanced",
                      C=C,
                      gamma=gamma,
                      random_state=__RANDOM_STATE__
                      )
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            print(f"Score for C={C}, gamma={gamma}: {score}")

            if score > best_score:
                best_model = svm
                best_score = score

    return best_model.get_params()


def fine_tune_log_reg(X_train, y_train, param_grid=param_grid, DEBUG=__DEBUG__, mode=None):
    """
    Fine-tune logistic regression model using GridSearchCV.
    Save fine-tuned model and vectorizer.
    Parameters:
        X_train: training features
        y_train: training labels
        param_grid: parameter grid
        DEBUG: debug flag
        mode: target of dataset
    """
    if DEBUG:
        return pol_logistic_parameters if mode == "political_leaning" else gen_logistic_parameters

    grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=500),
                                param_grid=param_grid,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=10
                                )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def X_with_pred_pol_lean(df, tfidf):
    X_post = tfidf.fit_transform(df['post'])
    X_pol = pd.get_dummies(df['predicted_political_leaning'], drop_first=True)
    X_combined = hstack([X_post, X_pol.values]).tocsr()
    return X_combined


def debug_X_with_pred_pol_lean(X, model):
    """
    Trim or add features such that dimensions of different X columns match.
    """
    print(f"Shape of X_combined: {X.shape}\nShape of model.coef_: {model.coef_.shape}")

    try:
        if X.shape[1] != model.coef_.shape[1]:
            print(f"Feature mismatch detected: X_combined has {X.shape[1]} features, but model expects {model.coef_.shape[1]} features.")
    except ValueError as e:
        print(f"Error: {e}\nAttempting to debug and fix feature mismatch...")
        feature_diff = X.shape[1] - model.coef_.shape[1]
        if feature_diff > 0:
            print(f"X_combined has {feature_diff} extra features. Trimming features...")
            X = X[:, :model.coef_.shape[1]]
        elif feature_diff < 0:
            print(f"X_combined is missing {-feature_diff} features. Adding zero-padding...")
            missing_features = -feature_diff
            zero_padding = csr_matrix((X.shape[0], missing_features))
            X = hstack([X, zero_padding]).tocsr()

    print(f"Fixed shape of X_combined: {X.shape}")
    return X.shape[1] == model.coef_.shape[1]
