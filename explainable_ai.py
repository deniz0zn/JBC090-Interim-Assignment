import pandas as pd
pd.set_option('future.no_silent_downcasting', True)  # remove future warning about .fillna()
import numpy as np
import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from config import __RANDOM_STATE__
from functools import reduce

### FOR DEBUG: N=10 BUT SHOULD BE N=100 IN FINAL!

class LimeEvaluator:
    """
    Class that evaluates the differences in features for predicted political leaning in each generation.
    :attributes: df, model, tfidf, random_state
    """
    def __init__(self, df: pd.DataFrame, model: LogisticRegression, tfidf: TfidfVectorizer, random_state=__RANDOM_STATE__) -> None:
        """
        Initializes a LimeEvaluator object.
        :param df: dataframe to sample from
        :param model: model used to predict political leaning
        :param tfidf: vectorizer used on features
        :param random_state: seed for reproducibility
        """
        self.df = df
        self.model = model
        self.tfidf = tfidf
        self.random_state = random_state

    def sample(self) -> pd.DataFrame:
        """
        Select a number of random rows from each generation.
        :return: sampled dataframe
        """
        df_sample = self.df.groupby("generation")[["post", "predicted_political_leaning",
                                                   "generation"]].apply(lambda x: x.sample(n=10, replace=True,
                                                                                           random_state=self.random_state))
        df_sample = df_sample.reset_index(drop=True)
        return df_sample

    def predict_proba_wrapper(self, texts: list[str]) -> np.ndarray:
        """
        Vectorize raw text and pass to predict_proba.
        :param texts: list of raw texts
        :return: vectorized texts
        """
        return self.model.predict_proba(self.tfidf.transform(texts))

    def explain(self) -> dict[str, pd.DataFrame]:
        """
        Initialize explainer and add most important features to each generation and political leaning.
        :return: dictionary of all dataframes conatining most important features.
        """
        explainer = LimeTextExplainer(class_names=self.model.classes_)
        generations = ['Baby boomers', 'Generation X', 'Millennials', 'Generation Z']
        leanings = ['left', 'center', 'right']
        dfs = {f"{g}_{l}": pd.DataFrame(columns=['Feature', 'Importance']) for g in generations for l in leanings}
        df_sample = self.sample()
        for row in df_sample.itertuples():
            explanation = explainer.explain_instance(row.post,
                                                     self.predict_proba_wrapper,
                                                     num_features=10
                                                     )
            df_explain = pd.DataFrame(explanation.as_list(), columns=['Feature', 'Importance'])
            key = f"{row.generation}_{row.predicted_political_leaning}"

            if key in dfs:
                dfs[key] = pd.merge(dfs[key], df_explain, on='Feature', how='outer', suffixes=('_df1', '_df2'))
                dfs[key]['Importance'] = dfs[key]['Importance_df1'].fillna(0) + dfs[key]['Importance_df2'].fillna(0)
                dfs[key] = dfs[key][['Feature', 'Importance']]

        return dfs

    @staticmethod
    def merge_features(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two dataframes together based on common features and sum importances.
        :param df1: dataframe 1 to merge
        :param df2: dataframe 2 to merge
        :return:merged dataframe.
        """
        combined = pd.merge(df1, df2, on='Feature', how='outer', suffixes=('_df1', '_df2'))
        combined['Importance'] = combined['Importance_df1'].fillna(0) + combined['Importance_df2'].fillna(0)
        return combined[['Feature', 'Importance']]

    @staticmethod
    def generation_unique(df_target: pd.DataFrame, dfs_other: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Find unique features for a specific dataframe compared to other dataframes.
        :param df_target: target dataframe to find unique features for
        :param dfs_other: other dataframe to compare feature uniqueness with
        :return:
        """
        all_features = set().union(*(df['Feature'] for df in dfs_other))
        unique_rows = df_target[~df_target['Feature'].isin(all_features)]
        return unique_rows

    def unique_all_generations(self, dfs: dict[str, pd.DataFrame]) -> None:
        """
        Print all unique features for each and all generations based on predicted political leaning.
        :param dfs: dictionary of dataframes for each generation and its political leaning
        """
        dfs_leaning_dict = {"left": [df for key, df in dfs.items() if "left" in key],
                            "center": [df for key, df in dfs.items() if "center" in key],
                            "right": [df for key, df in dfs.items() if "right" in key]}
        names = {0: "Baby boomers", 1: "Generation X", 2: "Millennials", 3: "Generation Z"}

        for key, dfs in dfs_leaning_dict.items():
            globals()[f"df_{key}"] = reduce(self.merge_features, dfs)
            globals()[f"df_{key}"] = globals()[f"df_{key}"].sort_values(by='Importance', ascending=False)
            print(f"\nUnique features from '{key}' from all generations:\n{globals()[f'df_{key}']}\n"
                  f"Unique features for each political leaning in each generation\n")
            
            unique_features_with_importance = {}
            for i, df_target in enumerate(dfs):
                dfs_others = dfs[:i] + dfs[i + 1:]
                df_unique = self.generation_unique(df_target, dfs_others)
                unique_features_with_importance[names[i]] = df_unique.sort_values(by='Importance', ascending=False)

            for df_name, unique_df in unique_features_with_importance.items():
                print(f"Unique features in {df_name} for '{key}':\n{unique_df.head(5)}\n")
