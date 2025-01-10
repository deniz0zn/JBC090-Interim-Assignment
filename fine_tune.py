import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from config import param_grid, __DEBUG__, svm_C_values, svm_gamma_values, __RANDOM_STATE__, test_size
from config_fine_tune import pol_logistic_parameters, gen_logistic_parameters, svm_parameters
from explainable_ai import Metrics


def fine_tune_svm(X_train, y_train, X_test, y_test, DEBUG=__DEBUG__) -> dict[str, any]:
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


def fine_tune_log_reg(X_train, y_train, param_grid=param_grid, DEBUG=__DEBUG__, mode=None) -> dict[str, any]:
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


def X_with_pred_pol_lean(df, tfidf, model):
    X_post = tfidf.transform(df['post'])
    X_pol = pd.get_dummies(df['predicted_political_leaning'], drop_first=True)
    X_combined = hstack([X_post, X_pol.values]).tocsr()
    print(f"Shape of X_combined: {X_combined.shape}\nShape of model.coef_: {model.coef_.shape}")
    right = True #False?

    while not right:
        try:
            if X_combined.shape[1] != model.coef_.shape[1]:
                print(
                    f"Feature mismatch detected: X_combined has {X_combined.shape[1]} features, but model expects {model.coef_.shape[1]} features.")
        except ValueError as e:
            print(f"Error: {e}\nAttempting to debug and fix feature mismatch...")
            feature_diff = X_combined.shape[1] - model.coef_.shape[1]
            if feature_diff > 0:
                print(f"X_combined has {feature_diff} extra features. Trimming features...")
                X_combined = X_combined[:, :model.coef_.shape[1]]
            elif feature_diff < 0:
                print(f"X_combined is missing {-feature_diff} features. Adding zero-padding...")
                missing_features = -feature_diff
                zero_padding = csr_matrix((X_combined.shape[0], missing_features))
                X_combined = hstack([X_combined, zero_padding]).tocsr()
                right = X_combined.shape[1] == model.coef_.shape[1]

    return X_combined


def run_fine_tuned_log_with_pol(X, df):
    y = df["generation"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=__RANDOM_STATE__, shuffle=True)
    parameters = fine_tune_log_reg(X_train=X_train, y_train=y_train, DEBUG=__DEBUG__, mode="generation")
    solver, penalty, C = parameters["solver"], parameters["penalty"], parameters["C"]
    model = LogisticRegression(class_weight="balanced", penalty=penalty, C=C, solver=solver, max_iter=1000,
                               random_state=__RANDOM_STATE__)
    print(f"Fitting the model with parameters: {model.get_params()}\n")
    model.fit(X_train, y_train)
    print(f"Model fitted. Metrics:\n{Metrics(X_test, y_test, model)}")
    return model
