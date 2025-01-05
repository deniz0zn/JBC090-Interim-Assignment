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


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])


class Dataset:
    """
    Class to handle data loading and provide brief information about the dataset.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)


    def rename_columns(self):
        if "birth_year" in self.data.columns:
            self.data = self.data.rename(columns={"auhtor_ID" : "author_id", "birth_year" : "label"})
        elif "political_leaning" in self.data.columns:
            self.data = self.data.rename(columns={"auhtor_ID": "author_id", "political_leaning" : "label"})


    def check_imbalance(self):
        class_distribution = self.data["target"].value_counts(normalize=True) * 100
        print("\nClass Distribution (%):")
        for label, percentage in class_distribution.items():
            print(f"{label}: {percentage:.2f}%")
        print(f"\nClass Distribution (Counts): \n{self.data["target"].value_counts()}")

    def word_count(self):
        self.data['word_count'] = self.data["post"].apply(lambda x: len(str(x).split()))
        return self.data['word_count'].mean()

    def char_count(self):
        self.data['char_count'] = self.data["post"].apply(lambda x: len(str(x)))
        return self.data['char_count'].mean()


    def gen(self, year: int) -> str:
        if 1946 <= year <= 1964:
            return 'Baby boomers'
        elif 1965 <= year <= 1980:
            return 'Generation X'
        elif 1981 <= year <= 1996:
            return 'Millennials'
        else:  # 1997 <= year <= 2012
            return 'Generation Z'

    def apply_gen(self) -> pd.DataFrame:
        self.data["generation"] = self.data["label"].apply(self.gen)


    def dataframe(self) -> pd.DataFrame:
        self.rename_columns()
        if "birth_year" in self.file_path:
            self.apply_gen()

        return self.data

    def __str__(self):
        self.data = self.dataframe()
        return(f"The dataset contains {len(self.data)} rows\n"
                f"The Columns of the dataset: {self.data.columns}\n"
                f"{self.check_imbalance()}\n"
                f"Average character count: {self.char_count()}"
                f"Average word count: {self.word_count()}"
                )




class DataCleaning:
    """
    Class to handle data cleaning for Phase 0.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        tqdm.pandas()

    def clean_text(self, text):

        with open("regex/birth_year.txt", 'r') as file:
            patterns = [line.strip() for line in file if line.strip()]
        for pattern in patterns:
            text = re.sub(pattern, "<AGE_TERM>", text)

        with open("regex/political_leaning.txt", 'r') as file:
            patterns = [line.strip() for line in file if line.strip()]
        for pattern in patterns:
            text = re.sub(pattern, "<POLITICAL_TERM>", text)

        # Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", "<URL>", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Normalize text
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in pos_tags]

    @staticmethod
    def _pos_tagger(t: str) -> str:
        """
        Convert wordnet POS to lemmatization POS.
        :param t: POS tag to convert
        :return: POS tag for lemmatization.
        """
        return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(t[0], wordnet.NOUN)


    def tokenize_data(self, df, text_column="post"):
        df_copy = df.copy()
        df_copy[text_column] = df_copy[text_column].progress_apply(self.tokenize_and_lemmatize)
        return df_copy


    def clean_data(self,df, text_column="post"):
        df_copy = df.copy()
        df_copy[text_column] = df_copy[text_column].progress_apply(lambda x: self.clean_text(x))
        return df_copy

class Reader:
    def __init__(self, path: str, tokenize: bool):
        self.dataloader =Dataset(path)
        self.df = self.dataloader.dataframe()
        self.cleaner = DataCleaning()
        self.tokenize = tokenize

    def dataset(self, tokenize: bool):
        print(str(self.dataloader))
        clean_data = self.cleaner.clean_data(self.df)

        return self.cleaner.tokenize_data(clean_data) if tokenize else clean_data

    def split_data(self, test_size: float, random_state: int):
        processed_df = self.dataset(self.tokenize)
        X = processed_df["post"]
        y = processed_df["label"]

        return train_test_split(X,y, test_size= test_size, random_state=random_state)




