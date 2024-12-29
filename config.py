__RANDOM_STATE__ = 42

param_grid = {
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'penalty': ['none', 'elasticnet', 'l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }