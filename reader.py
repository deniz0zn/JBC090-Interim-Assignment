import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
from config import __RANDOM_STATE__,test_size
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])


class Dataset:
    """
    Class to handle data loading and provide brief information about the dataset.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.target = "birth_year" if "birth_year" in self.df.columns else "political_leaning"

    # def rename_columns(self):
    #     if "birth_year" in self.data.columns:
    #         self.data = self.data.rename(columns={"auhtor_ID" : "author_id", "birth_year" : "label"})
    #     elif "political_leaning" in self.data.columns:
    #         self.data = self.data.rename(columns={"auhtor_ID": "author_id", "political_leaning" : "label"})

    def gen(self, year: int) -> str:
        if 1946 <= year <= 1964:
            return 'Baby boomers'
        elif 1965 <= year <= 1980:
            return 'Generation X'
        elif 1981 <= year <= 1996:
            return 'Millennials'
        else:  # 1997 <= year <= 2012
            return 'Generation Z'

    def apply_gen(self):
        if self.target == "birth_year":
            self.df["generation"] = self.df[self.target].apply(self.gen)
            self.target = "generation"

        return self.df

    def check_imbalance(self):
        class_distribution = self.df[self.target].value_counts(normalize=True) * 100
        print("\nClass Distribution (%):")
        for label, percentage in class_distribution.items():
            print(f"{label}: {percentage:.2f}%")
        print(f"\nClass Distribution (Counts):\n{self.df[self.target].value_counts()}\n")

    def word_count(self):
        self.df['word_count'] = self.df["post"].apply(lambda x: len(str(x).split()))
        return self.df['word_count'].mean()

    def char_count(self):
        self.df['char_count'] = self.df["post"].apply(lambda x: len(str(x)))
        return self.df['char_count'].mean()

    def __str__(self):
        self.apply_gen()
        return(f"The dataset contains {len(self.df)} rows\n"
                f"The columns of the dataset: {self.df.columns}\n"
                f"{self.check_imbalance()}\n"
                f"Average character count: {self.char_count():.3f}\n"
                f"Average word count: {self.word_count():.3f}"
                )


class DataCleaning:
    """
    Tokenize and clean a dataset.
    """
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _pos_tagger(t: str) -> str:
        """
        Convert wordnet POS to lemmatization POS.
        :param t: POS tag to convert
        :return: POS tag for lemmatization.
        """
        return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(t[0], wordnet.NOUN)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the sentence.
        :param text: text string to tokenize
        :return: list of tokens.
        """
        pos_text = pos_tag(word_tokenize(text))  # get POS for every word
        pos_tokens = [self.lemmatizer.lemmatize(word, self._pos_tagger(tag)).lower() for (word, tag) in pos_text]
        return pos_tokens

    def apply_tokenizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply tokenization to dataframe.
        :return: tokenized dataframes.
        """
        tqdm.pandas()  # show progress for each dataset
        df['post'] = df['post'].progress_apply(self.tokenize)
        df['post'] = df['post'].str.join(' ')
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove data pollution from dataframes e.g. ages and political terms.
        :return: cleaned dataframes.
        """
        if "birth_year" in df.columns:
            with open("regex/birth_year.txt", 'r') as file:
                patterns = [line.strip() for line in file if line.strip()]
            for pattern in patterns:
                df['post'] = df['post'].str.replace(pattern, '', regex=True)
        else:  # "political_leaning" in df.columns
            with open("regex/political_leaning.txt", 'r') as file:
                patterns = [line.strip() for line in file if line.strip()]
            for pattern in patterns:
                df['post'] = df['post'].str.replace(pattern, '<POLITICAL_TERM>', regex=True)

        return df


class Reader:
    def __init__(self, path: str, tokenize: bool):
        self.dataloader = Dataset(path)
        self.df = self.dataloader.apply_gen()
        self.tokenize = tokenize
        self.cleaner = DataCleaning()

    def dataset(self) -> pd.DataFrame:
        clean_data = self.cleaner.clean(self.df)
        return self.cleaner.apply_tokenizer(clean_data) if self.tokenize else clean_data

    def split_data(self, test_size=test_size, random_state=__RANDOM_STATE__):
        processed_df = self.dataset()
        X = processed_df["post"]

        if "birth_year" in processed_df.columns:
            y = processed_df["generation"]
        else:
            y = processed_df["political_leaning"]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# df_gen = Reader('datasets/birth_year.csv', tokenize=True).dataset()
