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


class DataLoader:
    """
    Class to handle data loading and provide brief information about the dataset.
    """
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.rename_columns()
        self.info()

    def info(self):
        print (f"The dataset contains {len(self.data)} rows\n"
                f"The Columns of the dataset: {self.data.columns}\n"
                f"{self.check_imbalance()}\n"
                f"Average character count: {self.char_count()}"
                f"Average word count: {self.word_count()}"
                )

    def rename_columns(self):
        if "birth_year" in self.data.columns:
            self.data = self.data.rename(columns={"auhtor_ID" : "author_id", "birth_year" : "target"})
        elif "political_leaning" in self.data.columns:
            self.data = self.data.rename(columns={"auhtor_ID": "author_id", "political_leaning" : "target"})


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




class DataCleaning:
    """
    Class to handle data cleaning for Phase 0.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        tqdm.pandas()  # Enable progress bar for pandas

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
    def get_wordnet_pos(treebank_tag):
        """
        Converts TreeBank POS tags to WordNet POS tags.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def process(self, df, text_column="post"):

        df_copy = df.copy()
        df_copy[text_column] = df_copy[text_column].progress_apply(lambda x: self.clean_text(x))
        df_copy[text_column] = df_copy[text_column].progress_apply(self.tokenize_and_lemmatize)
        return df_copy


class Reader:
    def __init__(self, path: str, split: bool):
        self.df = DataLoader(path)
        self.cleaner = DataCleaning()
        self.split = split
        self.train = None
        self.test = None

    def data_split(self, test_size=test_size, random_state=__RANDOM_STATE__):
        self.train, self.test = train_test_split(
            self.data, test_size=test_size, random_state=random_state, stratify=self.data[self.target_column]
        )
        return self.train, self.test

    def run(self):
        self.df = self.cleaner.process(self.df)

        if self.split:
            return self.data_split(test_size=self.test_size)

        return self.df

