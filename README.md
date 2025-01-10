# Combining Stylometry and NLP for Fair, Robust Reddit Author Profiling on Political Leaning and Generation

## Group 3
- Kaan Mutlu Çelik (1785559)
- Juliette Hattingh-Haasbroek (1779192)
- Deniz Özen (1734970)
- Berke Söker (1821458)


### Research Question
How can stylometry and NLP techniques be combined to develop robust, interpretable, and fair models for Reddit author profiling, specifically to predict a user’s generation based on their political views, while evaluating the explanatory power of political views in generation prediction?

# Overview
- [Paper Details](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#paper-details)
- [tl;dr](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#tldr)
- [Reproduction](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#reproduction)
- [Dependencies](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#dependencies)
- [Resources](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#resources)
- [Experimental Manupilation](https://github.com/deniz0zn/JBC090-Interim-Assignment/tree/main?tab=readme-ov-file#experimental-manipulation)

  
## Paper Details

## tl;dr

A tl;dr which highlights some points why someone who found your research code should care about this repository.

- Solved the Imbalance problem in both age and political leaning datasets by adjusting model weights.
- Trained and predicted each model on both datasets independently.
- Fine-tuned each model for better perfromance for each classification task.
- Used the models that were fine-tuned on political leaning dataset to predict the political views of the entries in the age dataset.
- Predicted the generation group again with the addition of the political leaning column.
- Checked the differences in metrics and employed explainable AI libraries to see if the addition of the political leaning column created any value for the classification task.
- Checked for robustness. (More on this later)

## Reproduction

1. Clone the repository:
   ```shell
   git clone https://github.com/deniz0zn/JBC090-Interim-Assignment.git 
   ```
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
3. Add the `birth_year.csv` and `political_leaning.csv` into the project directory and change the `PL_path` and `BY_path` in `config.py` file
   ```python
   PL_path = ["PATH OF POLITICAL LEANING DATASET"]
   BY_path = ["PATH OF BIRTH YEAR DATASET"]
4. Run the experiments:
   ```Shell
   python experiments.py
   ```

## Dependencies
The following code functions with the up-to-date version of the given libraries.

| Package         | Latest Version |
|-----------------|----------------|
| `pandas`        | 2.2.3          |
| `nltk`          | 3.8.1          |
| `tqdm`          | 4.66.1         |
| `scikit-learn`  | 1.6.0          |
| `joblib`        | 1.3.2          |
| `shap`          | 0.42.1         |
| `gensim`        | 4.3.1          |
| `numpy`         | 1.26.0         |
| `matplotlib`    | 3.8.0          |
| `lime`          | 0.2.0.1        |
| `fasttext`      | 0.9.2          |

## Resources

The total runtime for all computations was approximately 7 hours, 41 minutes, and 32 seconds on a system with an Intel i3-12300F CPU, GTX 1070 Ti GPU, and 16GB RAM.
Based on a power consumption of 0.108kW and the global average emission factor of 0.475 kg CO2 per kWh, the estimated CO2 emissions for this project are ~0.394 kg CO2.


## Experimental Manupilation
### General Parameters and Debug (config.py)
```Python
__RANDOM_STATE__ = 42
test_size = 0.3
__DEBUG__ = True
```
The `__RANDOM_STATE__` parameter controls the random state for reproducibility in dataset splitting and model initialization. Changing this value ensures consistent dataset splits for debugging and reproducibility. The `test_size` parameter adjusts the proportion of the dataset allocated to training and testing. Increasing the `test_size` reduces the training dataset size, impacting model training and evaluation reliability.

The `__DEBUG__` parameter controls whether the model uses pre-configured fine-tuned hyperparameters or enables runtime fine-tuning via grid search. When set to `True`, the model uses the pre-defined parameters from `config_fine_tune.py`. When set to `False`, it performs runtime fine-tuning, which may improve model performance but increases computation time.



### Fine-Tuning Parameters
```Python
# Preconfigured fine-tuned hyperparameters for Logistic Regression for each dataset
gen_logistic_parameters = {"penalty": "l2","C": 100,"solver" : "sag"}
pol_logistic_parameters = {"penalty": "l2","C": 100,"solver": "saga"}

# Preconfigured fine-tuned hyperparameters for Logistic Regression
svm_parameters = {"C": 100, "gamma": 0.1}
```

#### Logistic Regression (config_fine_tune.py)
The `penalty` parameter specifies the type of regularization applied to the model, with options including `l1`, `l2`, `elasticnet`, and `none`. Adjusting this parameter influences how the model handles overfitting. The `solver` parameter determines the optimization algorithm used for fitting the model, with options like `newton-cg`, `lbfgs`, `liblinear`, `sag`, and `saga`. Selecting different solvers can improve convergence or computation time based on the dataset size and characteristics. The `C` parameter represents the inverse of the regularization strength. Smaller values increase regularization, reducing the risk of overfitting.

#### SVM (config_fine_tune.py)
The `C` parameter controls the regularization strength for the SVM model. Higher values prioritize correctly classifying training data, which can reduce generalization to unseen data. The `gamma` parameter, which can take values such as `scale`, `auto`, or specific numerical values, determines the influence of a single training example on the decision boundary. Adjusting this parameter can refine the granularity of the decision boundary.

---


## Modular elements of the research code

Ideally: a section on how to add to the research code. Which components are modular and can be swapped out? How does one do that?

