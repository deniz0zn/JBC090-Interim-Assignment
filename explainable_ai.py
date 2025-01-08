import pandas as pd
import joblib
from lime.lime_text import LimeTextExplainer
from config import __RANDOM_STATE__
from functools import reduce

df = pd.read_parquet('datasets/birth_year_with_political_leaning.parquet')
model = joblib.load('models/fine_tuned_pol_model.pkl')
# tfidf = joblib.load('models/fine_tuned_pol_vectorizer.pkl')

# ADD LATER: UNIQUE VALUES BLABLABLA

class LimeEvaluator:
    """
    Evaluate differences in predicted political leaning for each generation.
    """
    def __init__(self, df, model, tfidf, random_state=__RANDOM_STATE__):
        self.df = df
        self.model = model
        self.tfidf = tfidf
        self.random_state = random_state

    def sample(self):
        df_sample = self.df.groupby('generation').apply(lambda x: x.sample(n=100, replace=True, random_state=__RANDOM_STATE__))
        df_sample = df_sample.reset_index(drop=True)
        return df_sample

    def predict_proba_wrapper(self, texts):
        """
        Vectorize raw text and pass to predict_proba.
        """
        return self.model.predict_proba(self.tfidf.transform(texts))

    def explain(self):
        explainer = LimeTextExplainer(class_names=self.model.classes_)
        generations = ['Baby boomers', 'Generation X', 'Millennials', 'Generation Z']
        leanings = ['left', 'center', 'right']
        dfs = {f"{g}_{l}": pd.DataFrame(columns=['Feature', 'Importance']) for g in generations for l in leanings}
        counters = {g: 1 for g in generations}
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
                print(f"Row {counters[row.generation]} done")
                counters[row.generation] += 1

        return dfs

    @staticmethod
    def merge_features(df1, df2):
        combined = pd.merge(df1, df2, on='Feature', how='outer', suffixes=('_df1', '_df2'))
        combined['Importance'] = combined['Importance_df1'].fillna(0) + combined['Importance_df2'].fillna(0)
        return combined[['Feature', 'Importance']]

    def unique_all_generations(self, dfs: dict[str, pd.DataFrame]):
        dfs_leaning_dict = {"left": [df for key, df in dfs.items() if "left" in key],
                            "center": [df for key, df in dfs.items() if "center" in key],
                            "right": [df for key, df in dfs.items() if "right" in key]}
        # dfs_left = [df for key, df in dfs.items() if 'left' in key]
        # dfs_center = [df for key, df in dfs.items() if 'center' in key]
        # dfs_right = [df for key, df in dfs.items() if 'right' in key]

        for key, dfs in dfs_leaning_dict.items():
            f"df_{key}" = reduce(self.merge_features, dfs)



# evaluator = LimeEvaluator(df, model, tfidf).explain()
