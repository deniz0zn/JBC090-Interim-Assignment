__RANDOM_STATE__ = 42
__DEBUG__ = True
param_grid = {
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'penalty': ['none', 'elasticnet', 'l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
CV = 5
test_size = 0.3

PL_PATH = "datasets/birth_year.csv"
BY_PATH = "datasets/political_leaning.csv"

svm_C_values = [1, 10, 100]
svm_gamma_values = ['scale', 'auto', 0.1, 0.01]