# Combining Stylometry and NLP for Fair, Robust Reddit Author Profiling on Political Leaning and Generation

## JBC090 Interim Assignment

### Research Question
ow can stylometry and NLP techniques be combined to develop robust, interpretable, and fair models for Reddit author profiling, specifically to predict a user’s generation based on their political views, while evaluating the explanatory power of political views in generation prediction?

# Overview

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

Awaiting the final pipeline from Deniz.

Instructions on how to reproduce the results in the paper (and how to get the data to do so), and what system it was built on (I generally provide Python version and OS, could be better, but it’s something).

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

## Resources

Input from kaan on how long the code takes to run with his GPU. (CO2 Emission should be calculated)

Resources required. What kind of CPU/GPU it was ran on, and how long that took. Bonus points if you calculate CO2 emissions.

## Configuration

Input from Deniz.

## Experimental Manupilation

Check in with the group

A section dedicated to experimental manipulation. What elements can be changed to change the experiment? Where do we change those? As you can see I even have specific line numbers in these (it’d probably be better if they were linked, but anyway).

## Modular elements of the research code

Ideally: a section on how to add to the research code. Which components are modular and can be swapped out? How does one do that?


## Group 3
- Kaan Mutlu Çelik (1785559)
- Juliette Hattingh-Haasbroek (1779192)
- Deniz Özen (1734970)
- Berke Söker (1821458)
