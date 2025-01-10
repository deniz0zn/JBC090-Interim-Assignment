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
from nltk.data import find
from config import __RANDOM_STATE__,test_size

nltk_resources = {
    "punkt": 'tokenizers/punkt',
    "wordnet": 'corpora/wordnet',
    "averaged_perceptron_tagger": 'taggers/averaged_perceptron_tagger',
    "stopwords": 'corpora/stopwords',
    "omw-1.4": 'corpora/omw-1.4',
}
for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"{resource_name} not found. Downloading...")
        nltk.download(resource_name)


class Dataset:
    """
    Loads the data, provides a brief description of the dataset.
    If dataset contains the `birth_year` column, add `generation` column.
    :attribute: file_path
    """
    def __init__(self, file_path: str) -> None:
        """
        Initializes a Dataset object.
        :param file_path:
        """
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.target = "birth_year" if "birth_year" in self.df.columns else "political_leaning"

    def gen(self, year: int) -> str:
        """
        Determine generation depending on given year.
        :param year: year to determine generation
        :return: generation of year.
        """
        if 1946 <= year <= 1964:
            return 'Baby boomers'
        elif 1965 <= year <= 1980:
            return 'Generation X'
        elif 1981 <= year <= 1996:
            return 'Millennials'
        else:  # 1997 <= year <= 2012
            return 'Generation Z'

    def apply_gen(self) -> pd.DataFrame:
        """
        Add `generation` column.
        :return: dataframe with `generation` column.
        """
        if self.target == "birth_year":
            self.df["generation"] = self.df[self.target].apply(self.gen)
            self.target = "generation"

        return self.df

    def check_imbalance(self) -> None:
        """
        Shows class imbalance of target in a dataset.
        """
        class_distribution = self.df[self.target].value_counts(normalize=True) * 100
        print("\nClass Distribution (%):")
        for label, percentage in class_distribution.items():
            print(f"{label}: {percentage:.2f}%")
        print(f"\nClass Distribution (Counts):\n{self.df[self.target].value_counts()}\n")

    def word_count(self) -> float:
        """
        Determine mean word count of `post` column.
        :return: average word count.
        """
        self.df['word_count'] = self.df["post"].apply(lambda x: len(str(x).split()))
        return self.df['word_count'].mean()

    def char_count(self) -> float:
        """
        Determine mean character count of `post` column.
        :return: average character count.
        """
        self.df['char_count'] = self.df["post"].apply(lambda x: len(str(x)))
        return self.df['char_count'].mean()

    def __str__(self) -> str:
        """
        If `birth_year` in columns of dataframe, add `generation` column.
        Return string representation of Dataset object.
        :return: string representation of Dataset object.
        """
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
        """
        Initializes a DataCleaning object.
        """
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
        :param df: dataframe to tokenize
        :return: tokenized dataframe.
        """
        print("\nTokenizing dataset.")
        tqdm.pandas()  # show progress for each dataset
        df['post'] = df['post'].progress_apply(self.tokenize)
        df['post'] = df['post'].str.join(' ')
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove data pollution from dataframes e.g. ages and political terms.
        :param df: dataframe to clean
        :return: cleaned dataframe.
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
    """
    Clean, tokenize, and split data.
    :attributes: path, tokenize
    """
    def __init__(self, path: str, tokenize: bool) -> None:
        """
        Initializes a Reader object.
        :param path: path read dataset
        :param tokenize: flag to tokenize dataset
        """
        self.dataloader = Dataset(path)
        self.target = self.dataloader.target
        self.df = self.dataloader.apply_gen()
        self.tokenize = tokenize
        self.cleaner = DataCleaning()

    def dataset(self) -> pd.DataFrame:
        """
        Clean dataset, and tokenize if flagged.
        :return: Cleaned (and tokenized) dataset.
        """
        clean_data = self.cleaner.clean(self.df)
        return self.cleaner.apply_tokenizer(clean_data) if self.tokenize else clean_data

    def split_data(self, test_size=test_size, random_state=__RANDOM_STATE__):
        """
        Split data into training and test set.
        :param test_size: size of test set
        :param random_state: seed for reproducibility
        :return: train and test sets.
        """
        processed_df = self.dataset()
        X = processed_df["post"]

        if "birth_year" in processed_df.columns:
            y = processed_df["generation"]
        else:
            y = processed_df["political_leaning"]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)
